"""Binance spot klines adapter.

The first adapter against a real venue. Everything specific to Binance lives here; the
mapping, availability, and duplicate handling reuse
:mod:`qresearch.data.adapters.base`.

Timestamp label: ``bar_start``, established empirically rather than from documentation.
A kline's ``field[0]`` was compared against the raw ``aggTrades`` in
``[field[0], field[0] + 60s)`` for BTCUSDT at 2024-03-04 12:00Z: open, high, low, and
close all matched the first / max / min / last trade in that window exactly. The proof is
frozen in ``tests/contracts/test_binance_contract.py``; ``field[6]`` is the inclusive
close (one millisecond short of the next open) and is used only as a consistency check.

Availability: klines are served as soon as the interval closes -- a measurement at two
minute boundaries found the bar available 1.1-1.4 s after close, entirely accounted for
by network round trip. So publication latency here is a property of *your collection
setup* (poll interval plus RTT), not of the venue, and it is a required, documented
policy value rather than a guess baked into the adapter. A live recorder that stamps real
receipt times should supply ``available_at`` directly instead.

Only closed bars are returned. A kline whose interval has not ended yet carries partial
OHLCV, and ingesting one would write a bar that later changes -- the adapter drops it.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

import polars as pl

from qresearch.data.adapters.base import (
    IngestRequest,
    NormalizedBatch,
    apply_duplicate_policy,
    derive_bar_bounds,
)
from qresearch.data.contracts import AssetClass, Instrument, SymbolAlias
from qresearch.data.manifests import SourceRef, TimestampLabel
from qresearch.ids import InstrumentId
from qresearch.logging import get_logger
from qresearch.time import UTC, ensure_utc, now_utc, parse_duration

log = get_logger(__name__)

SOURCE_NAME: Final = "binance:spot-klines-v3"
DATA_API: Final = "https://data-api.binance.vision"
"""Binance's public market-data host. No API key, no account, and it is the endpoint
Binance itself points at for historical data."""

MAX_LIMIT: Final = 1000
SCHEME: Final = "binance"

_INTERVALS: Final[dict[str, str]] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
}

# Kline array positions (see the module docstring for the empirical check).
_OPEN_TIME, _OPEN, _HIGH, _LOW, _CLOSE, _VOLUME, _CLOSE_TIME = 0, 1, 2, 3, 4, 5, 6
_QUOTE_VOLUME, _TRADES = 7, 8


class BinanceError(RuntimeError):
    """The venue returned something the adapter will not guess about."""


@dataclass(frozen=True, slots=True)
class BinanceQuery:
    """A parsed ``binance://`` URI.

    Spelling: ``binance://spot?interval=1m&start=2024-03-04T00:00:00Z&end=...``. The URI
    is the provenance string recorded on the dataset, so it must describe the pull
    completely -- symbols come from the request's instrument aliases.
    """

    market: str
    interval: str
    start: _dt.datetime
    end: _dt.datetime

    @classmethod
    def parse(cls, uri: str) -> BinanceQuery:
        parts = urllib.parse.urlparse(uri)
        if parts.scheme != SCHEME:
            raise ValueError(f"expected a {SCHEME}:// uri, got {uri!r}")
        market = parts.netloc or "spot"
        if market != "spot":
            raise ValueError(f"only the spot market is supported, got {market!r}")
        params = urllib.parse.parse_qs(parts.query)
        missing = [k for k in ("interval", "start", "end") if k not in params]
        if missing:
            raise ValueError(
                f"binance uri is missing {missing}; expected "
                f"{SCHEME}://spot?interval=1m&start=<iso8601>&end=<iso8601>"
            )
        start = ensure_utc(_dt.datetime.fromisoformat(params["start"][0]))
        end = ensure_utc(_dt.datetime.fromisoformat(params["end"][0]))
        if start >= end:
            raise ValueError(f"start {start} must precede end {end}; the range is half-open")
        return cls(market=market, interval=params["interval"][0], start=start, end=end)

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        return int(self.end.timestamp() * 1000)


def build_uri(
    *, interval: str, start: _dt.datetime, end: _dt.datetime, market: str = "spot"
) -> str:
    """Canonical ``binance://`` URI for a pull."""
    query = urllib.parse.urlencode(
        {
            "interval": interval,
            "start": ensure_utc(start).isoformat().replace("+00:00", "Z"),
            "end": ensure_utc(end).isoformat().replace("+00:00", "Z"),
        }
    )
    return f"{SCHEME}://{market}?{query}"


class BinanceKlineAdapter:
    """Fetches closed spot klines over the public REST endpoint.

    Args:
        base_url: host to call. Defaults to Binance's public data host.
        max_retries: attempts per page on 429/5xx, with exponential backoff honouring
            ``Retry-After`` when present.
        pace: seconds to wait between pages, to stay well inside the weight budget.
        now: clock used to decide which bars have closed; injectable for tests.
    """

    def __init__(
        self,
        *,
        base_url: str = DATA_API,
        max_retries: int = 4,
        pace: float = 0.12,
        timeout: float = 20.0,
        now: Any = now_utc,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.pace = pace
        self.timeout = timeout
        self._now = now

    # -- adapter protocol ------------------------------------------------------------

    def read(self, request: IngestRequest) -> NormalizedBatch:
        query = BinanceQuery.parse(request.uri)
        self._check_interval(query, request)
        if request.policy.timestamp_label is not TimestampLabel.BAR_START:
            raise ValueError(
                "Binance klines are labelled by the interval start; the policy declares "
                f"{request.policy.timestamp_label.value!r}. The label is verified against "
                "raw trades in tests/contracts/test_binance_contract.py."
            )

        symbols = self._symbols(request, query)
        frames: list[pl.DataFrame] = []
        digest = hashlib.sha256()
        dropped: list[str] = []
        closed_before = self._closed_before(query)

        for instrument_id, symbol in sorted(symbols.items()):
            raw = list(self._paginate(symbol, query))
            # Hash the raw venue payload, so re-pulling an identical window resolves to
            # the identical dataset id.
            digest.update(symbol.encode())
            digest.update(json.dumps(raw, separators=(",", ":"), sort_keys=True).encode())
            usable = [k for k in raw if int(k[_CLOSE_TIME]) < closed_before]
            if len(usable) != len(raw):
                dropped.append(f"{symbol}:{len(raw) - len(usable)} unclosed")
            if not usable:
                log.warning("no closed bars for %s in the requested window", symbol)
                continue
            frames.append(self._to_frame(usable, instrument_id=instrument_id, request=request))
            log.info(
                "fetched %s",
                symbol,
                extra={
                    "fields": {
                        "symbol": symbol,
                        "bars": len(usable),
                        "dropped": len(raw) - len(usable),
                    }
                },
            )

        if not frames:
            raise BinanceError(
                f"no closed {query.interval} bars for {sorted(symbols.values())} in "
                f"[{query.start}, {query.end})"
            )
        frame = pl.concat(frames)
        frame = apply_duplicate_policy(frame, request.policy.duplicates)
        frame = frame.sort(["instrument_id", "bar_start", "revision"])
        return NormalizedBatch(
            frame=frame,
            source=SourceRef(
                provider=request.source_name,
                feed=request.feed or f"{query.market}/{query.interval}",
                uri=request.uri,
                content_sha256=digest.hexdigest(),
                source_revision=request.source_revision,
            ),
            dropped_unmapped_symbols=tuple(dropped),
        )

    # -- fetching ---------------------------------------------------------------------

    def _paginate(self, symbol: str, query: BinanceQuery) -> Iterator[list[Any]]:
        """Yield raw klines for ``symbol``, walking ``startTime`` forward.

        Binance returns at most ``MAX_LIMIT`` per call and the window is inclusive of
        ``startTime``, so each page resumes one millisecond past the last open time.
        """
        cursor = query.start_ms
        seen: set[int] = set()
        while cursor < query.end_ms:
            page = self._get(
                "/api/v3/klines",
                symbol=symbol,
                interval=query.interval,
                startTime=cursor,
                endTime=query.end_ms - 1,
                limit=MAX_LIMIT,
            )
            if not page:
                return
            for kline in page:
                open_ms = int(kline[_OPEN_TIME])
                if open_ms in seen or open_ms >= query.end_ms:
                    continue
                seen.add(open_ms)
                yield kline
            last_open = int(page[-1][_OPEN_TIME])
            if last_open + 1 <= cursor:
                return  # no forward progress; stop rather than loop
            cursor = last_open + 1
            if len(page) < MAX_LIMIT:
                return
            time.sleep(self.pace)

    def _fetch(self, url: str) -> Any:
        """The only place the network is touched. Stubbed in tests."""
        request = urllib.request.Request(url, headers={"User-Agent": "qresearch"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _get(self, path: str, **params: Any) -> list[Any]:
        """Fetch one page, retrying only what is worth retrying.

        429/418 (rate limited) and 5xx are the venue's problem and back off; a 4xx is our
        bug and is surfaced immediately rather than hammering the endpoint.
        """
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(params)}"
        delay = 0.5
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = self._fetch(url)
                if not isinstance(payload, list):
                    raise BinanceError(f"expected a list from {path}, got {type(payload).__name__}")
                return payload
            except urllib.error.HTTPError as error:
                retryable = error.code in (418, 429) or 500 <= error.code < 600
                if not retryable or attempt == self.max_retries:
                    raise BinanceError(
                        f"{path} failed with HTTP {error.code}: {error.reason}"
                    ) from error
                wait = float(error.headers.get("Retry-After") or delay) if error.headers else delay
                log.warning(
                    "binance %s (attempt %d/%d), waiting %.1fs",
                    error.code,
                    attempt,
                    self.max_retries,
                    wait,
                )
                time.sleep(wait)
                delay *= 2
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                if attempt == self.max_retries:
                    raise BinanceError(
                        f"{path} failed after {attempt} attempts: {error}"
                    ) from error
                time.sleep(delay)
                delay *= 2
        raise BinanceError(f"{path} exhausted retries")

    # -- normalization -----------------------------------------------------------------

    def _to_frame(
        self, klines: Sequence[Any], *, instrument_id: str, request: IngestRequest
    ) -> pl.DataFrame:
        step = parse_duration(request.bar_size)
        frame = pl.DataFrame(
            {
                "_source_ts": [
                    _dt.datetime.fromtimestamp(int(k[_OPEN_TIME]) / 1000, UTC) for k in klines
                ],
                "_close_ms": [int(k[_CLOSE_TIME]) for k in klines],
                "open": [float(k[_OPEN]) for k in klines],
                "high": [float(k[_HIGH]) for k in klines],
                "low": [float(k[_LOW]) for k in klines],
                "close": [float(k[_CLOSE]) for k in klines],
                "volume": [float(k[_VOLUME]) for k in klines],
                "quote_volume": [float(k[_QUOTE_VOLUME]) for k in klines],
                "trade_count": [int(k[_TRADES]) for k in klines],
            },
            schema_overrides={"_source_ts": pl.Datetime("us", "UTC")},
        )
        frame = derive_bar_bounds(
            frame,
            bar_size=request.bar_size,
            label=TimestampLabel.BAR_START,
            timestamp_column="_source_ts",
        )
        self._check_intervals(frame, step)
        latency = request.policy.publication_latency // _dt.timedelta(microseconds=1)
        return frame.with_columns(
            pl.lit(instrument_id).alias("instrument_id"),
            pl.lit(request.bar_size).alias("bar_size"),
            (pl.col("bar_end") + pl.duration(microseconds=latency)).alias("available_at"),
            # Volume-weighted average price, exact from the venue's own quote volume.
            pl.when(pl.col("volume") > 0)
            .then(pl.col("quote_volume") / pl.col("volume"))
            .otherwise(None)
            .alias("vwap"),
            pl.lit(request.source_name).alias("source"),
            (
                pl.lit(f"{instrument_id}:") + pl.col("_source_ts").dt.epoch("ms").cast(pl.String)
            ).alias("source_key"),
            pl.lit(request.source_revision, dtype=pl.Int32).alias("revision"),
            pl.lit(self._now()).cast(pl.Datetime("us", "UTC")).alias("ingested_at"),
        ).select(
            "instrument_id",
            "bar_size",
            "bar_start",
            "bar_end",
            "available_at",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
            "source",
            "source_key",
            "revision",
            "ingested_at",
        )

    @staticmethod
    def _check_intervals(frame: pl.DataFrame, step: _dt.timedelta) -> None:
        """Cross-check the venue's own close time against the derived ``bar_end``.

        ``field[6]`` is inclusive (one millisecond before the next open), so a consistent
        kline satisfies ``close_ms + 1 == bar_end``. A mismatch means the interval is not
        what the adapter assumes, and ingesting it would shift the dataset in time.
        """
        expected = frame.get_column("bar_end").dt.epoch("ms")
        actual = frame.get_column("_close_ms") + 1
        bad = (expected != actual).sum()
        if bad:
            first = frame.filter(expected != actual).row(0, named=True)
            raise BinanceError(
                f"{bad} kline(s) have a close time inconsistent with a {step} interval; "
                f"first at {first['bar_start']}. The venue's interval is not what this "
                "adapter assumes -- refusing to write a dataset that may be shifted in time."
            )

    def _closed_before(self, query: BinanceQuery) -> int:
        """Epoch ms past which a bar cannot yet have closed."""
        return min(query.end_ms, int(self._now().timestamp() * 1000))

    @staticmethod
    def _check_interval(query: BinanceQuery, request: IngestRequest) -> None:
        expected = _INTERVALS.get(request.bar_size)
        if expected is None:
            raise ValueError(
                f"bar_size {request.bar_size!r} has no Binance interval; "
                f"supported: {sorted(_INTERVALS)}"
            )
        if query.interval != expected:
            raise ValueError(
                f"uri interval {query.interval!r} disagrees with bar_size {request.bar_size!r}"
            )

    @staticmethod
    def _symbols(request: IngestRequest, query: BinanceQuery) -> dict[str, str]:
        """Instrument id -> venue symbol, resolved at the window start.

        Uses the same effective-dated aliases as every other adapter, so a ticker change
        resolves point-in-time rather than being rewritten by today's mapping.
        """
        out: dict[str, str] = {}
        for instrument in request.instruments:
            symbol = instrument.symbol_at(query.start, source=request.source_name)
            if symbol is not None:
                out[str(instrument.instrument_id)] = symbol
        if not out:
            raise ValueError(
                f"no instrument has a {request.source_name!r} alias effective at {query.start}; "
                "the venue symbol cannot be resolved"
            )
        return out


def fetch_instruments(
    symbols: Sequence[str],
    *,
    base_url: str = DATA_API,
    venue: str = "BINANCE",
    calendar_id: str = "24x7:1",
    listed_from: _dt.datetime | None = None,
    timeout: float = 20.0,
) -> tuple[Instrument, ...]:
    """Build :class:`Instrument` definitions from the venue's own ``exchangeInfo``.

    Tick size, step size, and base/quote assets are facts the venue publishes; hand-writing
    them into a config is tedious and easy to get subtly wrong -- a wrong
    ``quantity_increment`` silently changes every order size in a backtest.

    ``listed_from`` dates the symbol aliases. It defaults to well before Binance existed,
    which is right for a fixed universe but wrong if you care about when a pair actually
    listed; supply it explicitly for a point-in-time universe.
    """
    wanted = [s.upper() for s in symbols]
    url = f"{base_url.rstrip('/')}/api/v3/exchangeInfo"
    request = urllib.request.Request(url, headers={"User-Agent": "qresearch"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    by_symbol = {s["symbol"]: s for s in payload.get("symbols", [])}
    missing = [s for s in wanted if s not in by_symbol]
    if missing:
        raise BinanceError(f"unknown symbols on this venue: {missing}")

    effective_from = listed_from or _dt.datetime(2015, 1, 1, tzinfo=UTC)
    out = []
    for symbol in wanted:
        info = by_symbol[symbol]
        if info.get("status") != "TRADING":
            log.warning("%s is not TRADING (status=%s)", symbol, info.get("status"))
        filters = {f["filterType"]: f for f in info.get("filters", [])}
        tick = filters.get("PRICE_FILTER", {}).get("tickSize", "0.01")
        step = filters.get("LOT_SIZE", {}).get("stepSize", "0.00000001")
        out.append(
            Instrument(
                instrument_id=InstrumentId(f"CRYPTO:{symbol}"),
                asset_class=AssetClass.CRYPTO,
                venue=venue,
                quote_currency=info["quoteAsset"],
                base_currency=info["baseAsset"],
                price_increment=Decimal(str(tick)).normalize(),
                quantity_increment=Decimal(str(step)).normalize(),
                calendar_id=calendar_id,
                aliases=(
                    SymbolAlias(symbol=symbol, source=SOURCE_NAME, effective_from=effective_from),
                ),
            )
        )
    return tuple(out)
