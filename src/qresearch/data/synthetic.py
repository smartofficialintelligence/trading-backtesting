"""Deterministic synthetic bar data for tests, examples, and demos.

DEVELOPMENT_PLAN.md Stage 1 asks for "a tiny synthetic dataset containing gaps, delayed
observations, duplicate source rows, and multiple assets". Those four properties are the
awkward cases that a clean fixture would hide, and every one of them corresponds to a real
failure mode: a gap breaks rolling windows, a delayed bar breaks the assumption that
availability tracks bar_end, duplicates force the revision policy to matter, and a second
asset exposes any hidden dependence on iteration order.

Output is the *source* CSV format rather than canonical bars, so tests exercise the whole
adapter path including the timestamp-label convention.

Generation is seeded and uses only the standard library's ``random``, so a given seed
yields identical prices on every platform and Python build.
"""

from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from random import Random

from qresearch.data.calendars import get_calendar
from qresearch.data.contracts import AssetClass, Instrument, SymbolAlias
from qresearch.ids import InstrumentId
from qresearch.time import UTC

SOURCE_NAME = "synthetic:bars-v1"
SOURCE_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume", "trades")


@dataclass(frozen=True, slots=True)
class SyntheticAsset:
    symbol: str
    instrument_id: str
    start_price: float
    volatility: float
    """Per-minute lognormal-ish step size as a fraction of price."""

    base_volume: float


DEFAULT_ASSETS: tuple[SyntheticAsset, ...] = (
    SyntheticAsset("BTC-USD", "CRYPTO:BTCUSD", 42_000.0, 0.0008, 12.0),
    SyntheticAsset("ETH-USD", "CRYPTO:ETHUSD", 2_300.0, 0.0011, 180.0),
)

EQUITY_ASSETS: tuple[SyntheticAsset, ...] = (
    SyntheticAsset("ACME", "EQ:ACME", 150.0, 0.0004, 20_000.0),
    SyntheticAsset("GLBX", "EQ:GLBX", 48.0, 0.0006, 55_000.0),
)


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """What the generated fixture should contain."""

    start: _dt.datetime = _dt.datetime(2024, 3, 4, 0, 0, tzinfo=UTC)
    minutes: int = 240
    assets: tuple[SyntheticAsset, ...] = DEFAULT_ASSETS
    seed: int = 20240304

    gap_minutes: tuple[int, ...] = (61, 62, 63)
    """Minute offsets absent for the *second* asset only, so the two assets disagree about
    which timestamps exist -- the situation that reveals cross-sectional asynchrony."""

    delayed_minutes: tuple[int, ...] = (30, 150)
    """Minute offsets published late (see :attr:`delay`), for the first asset only."""

    delay: _dt.timedelta = _dt.timedelta(seconds=90)
    """Publication delay for a "late" bar.

    Ninety seconds is the smallest value that exceeds one bar interval, which is what
    makes availability run *backwards*: bar N+1 publishes before the late bar N, so a
    point-in-time read returns a hole in the middle of the series rather than a truncated
    tail. That inversion is the case worth testing, and nothing shorter produces it.

    It is deliberately not minutes. Historical REST ingestion usually has a *constant*
    publication latency, and real lateness, when it happens, is seconds; a fixture baking
    in a long delay would invite pessimistic defaults elsewhere. This is a plausible bad
    hiccup, not a norm. Tests needing a larger hole pass an explicit ``delay`` and say
    why."""
    duplicate_minutes: tuple[int, ...] = (10,)
    """Minute offsets emitted twice, the second copy at a higher revision."""

    normal_latency: _dt.timedelta = _dt.timedelta(seconds=2)
    calendar_id: str = "24x7:1"
    """Bars are generated only inside this calendar's sessions."""

    asset_class: AssetClass = AssetClass.CRYPTO
    venue: str = "SYNTH"

    @classmethod
    def equity(cls) -> SyntheticSpec:
        """Two US equities over two XNYS sessions (2024-03-04 and -05, EST)."""
        return cls(
            start=_dt.datetime(2024, 3, 4, 14, 0, tzinfo=UTC),
            minutes=2 * 24 * 60,
            assets=EQUITY_ASSETS,
            seed=20240305,
            gap_minutes=(90, 91),
            delayed_minutes=(45, 400),
            duplicate_minutes=(12,),
            calendar_id="XNYS:1",
            asset_class=AssetClass.EQUITY,
            venue="XNYS",
        )


def instruments_for(spec: SyntheticSpec) -> tuple[Instrument, ...]:
    """Instrument definitions matching a spec's assets."""
    crypto = spec.asset_class is AssetClass.CRYPTO
    return tuple(
        Instrument(
            instrument_id=InstrumentId(asset.instrument_id),
            asset_class=spec.asset_class,
            venue=spec.venue,
            quote_currency="USD",
            base_currency=asset.symbol.split("-")[0] if crypto else None,
            price_increment=Decimal("0.01"),
            quantity_increment=Decimal("0.00000001") if crypto else Decimal("1"),
            calendar_id=spec.calendar_id,
            aliases=(
                SymbolAlias(
                    symbol=asset.symbol,
                    source=SOURCE_NAME,
                    effective_from=spec.start - _dt.timedelta(days=365),
                ),
            ),
        )
        for asset in spec.assets
    )


@dataclass
class _Row:
    timestamp: _dt.datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    available_at: _dt.datetime
    revision: int = 0
    extra: dict[str, str] = field(default_factory=dict)


def generate_rows(spec: SyntheticSpec) -> list[_Row]:
    """Generate source rows, timestamped by ``bar_start``."""
    rows: list[_Row] = []
    calendar = get_calendar(spec.calendar_id)
    for index, asset in enumerate(spec.assets):
        rng = Random(spec.seed + index * 1_000)
        price = asset.start_price
        for minute in range(spec.minutes):
            bar_start = spec.start + _dt.timedelta(minutes=minute)
            bar_end = bar_start + _dt.timedelta(minutes=1)
            if calendar.session_at(bar_start) is None:
                continue  # venue closed: no bar exists

            drift = rng.gauss(0.0, asset.volatility)
            open_price = price
            close_price = round(open_price * (1.0 + drift), 2)
            wick = abs(rng.gauss(0.0, asset.volatility)) * open_price
            high = round(max(open_price, close_price) + wick, 2)
            low = round(min(open_price, close_price) - wick, 2)
            low = max(low, 0.01)
            volume = round(asset.base_volume * (0.4 + rng.random()), 4)
            price = close_price

            if index == 1 and minute in spec.gap_minutes:
                continue  # a genuine hole: no bar was produced for this interval

            late = index == 0 and minute in spec.delayed_minutes
            available_at = bar_end + (spec.delay if late else spec.normal_latency)

            rows.append(
                _Row(
                    timestamp=bar_start,
                    symbol=asset.symbol,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                    trades=int(volume * 3) + 1,
                    available_at=available_at,
                )
            )

            if index == 0 and minute in spec.duplicate_minutes:
                # A provider correction: same interval, higher revision, slightly revised
                # close, published later than the original.
                rows.append(
                    _Row(
                        timestamp=bar_start,
                        symbol=asset.symbol,
                        open=open_price,
                        high=high,
                        low=low,
                        close=round(close_price * 1.0001, 2),
                        volume=volume,
                        trades=int(volume * 3) + 1,
                        available_at=available_at + _dt.timedelta(minutes=3),
                        revision=1,
                    )
                )
    rows.sort(key=lambda r: (r.symbol, r.timestamp, r.revision))
    return rows


def write_source_csv(
    path: Path | str, spec: SyntheticSpec | None = None, *, include_availability: bool = True
) -> Path:
    """Write a synthetic source CSV and return its path.

    Args:
        include_availability: emit ``available_at`` and ``revision`` columns. When False
            the file looks like a plain OHLCV export and availability must be derived from
            the policy's publication latency instead -- which is the common real case.
    """
    spec = spec or SyntheticSpec()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = list(SOURCE_COLUMNS)
    if include_availability:
        columns += ["available_at", "revision"]

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in generate_rows(spec):
            record = [
                _iso(row.timestamp),
                row.symbol,
                f"{row.open:.2f}",
                f"{row.high:.2f}",
                f"{row.low:.2f}",
                f"{row.close:.2f}",
                f"{row.volume:.4f}",
                str(row.trades),
            ]
            if include_availability:
                record += [_iso(row.available_at), str(row.revision)]
            writer.writerow(record)
    return target


def _iso(value: _dt.datetime) -> str:
    """ISO-8601 with an explicit ``Z``; the adapter rejects offset-free timestamps."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
