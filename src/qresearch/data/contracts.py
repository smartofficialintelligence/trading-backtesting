"""Canonical market and reference data contracts.

Implements the ``Instrument`` and ``Bar`` definitions in ARCHITECTURE.md sec. 4 and 6.
The invariants asserted here are the first line of defence against look-ahead bias: a
``Bar`` that cannot represent an availability time earlier than its own interval end
cannot be used to trade on information that did not exist yet.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from qresearch.config import FrozenModel
from qresearch.ids import InstrumentId
from qresearch.time import UtcDatetime, parse_duration

Symbol = Annotated[str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)]
BarSize = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*[smhd]$")]
"""Canonical bar duration such as ``1m`` or ``5m`` (see :func:`qresearch.time.parse_duration`)."""

Currency = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3,10}$")]
"""ISO-4217 for fiat; uppercase ticker for crypto assets that have no ISO code."""


class AssetClass(StrEnum):
    EQUITY = "equity"
    CRYPTO = "crypto"


class PriceAdjustment(StrEnum):
    """How prices in a dataset relate to corporate actions.

    Recorded on the dataset manifest, never inferred. Mixing adjusted signals with
    unadjusted execution prices is a listed leakage risk (ARCHITECTURE.md sec. 9).
    """

    NONE = "none"
    """Raw as-traded prices."""

    SPLIT_ONLY = "split_only"
    TOTAL_RETURN = "total_return"
    """Split and cash-distribution adjusted."""

    NOT_APPLICABLE = "not_applicable"
    """Crypto and other instruments with no corporate-action concept."""


class SymbolAlias(FrozenModel):
    """A provider-facing symbol, valid over a half-open UTC interval.

    Effective dating is what keeps a ticker change from rewriting history: a query for
    2019 resolves the symbol that was in use in 2019.
    """

    symbol: Symbol
    source: str = Field(min_length=1)
    effective_from: UtcDatetime
    effective_to: UtcDatetime | None = None

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError(
                f"symbol alias {self.symbol!r} has effective_to <= effective_from "
                f"({self.effective_to} <= {self.effective_from}); intervals are half-open "
                "[effective_from, effective_to) and must be non-empty"
            )
        return self

    def covers(self, moment: _dt.datetime) -> bool:
        """Whether this alias is the valid symbol at ``moment``."""
        if moment < self.effective_from:
            return False
        return self.effective_to is None or moment < self.effective_to


class Instrument(FrozenModel):
    """Stable identity for a tradable asset, independent of any provider's ticker."""

    instrument_id: InstrumentId
    asset_class: AssetClass
    venue: str = Field(min_length=1)
    quote_currency: Currency
    """Currency the price is denominated in (USD for AAPL, USD for BTC-USD)."""

    base_currency: Currency | None = None
    """For crypto pairs, the asset being priced (BTC in BTC-USD). None for equities."""

    price_increment: Decimal = Field(gt=0)
    """Minimum price tick. Decimal because tick alignment is an exactness question."""

    quantity_increment: Decimal = Field(gt=0)
    """Minimum tradable size increment. 1 for whole-share equities."""

    calendar_id: str = Field(min_length=1)
    """Versioned trading-calendar identifier, e.g. ``XNYS:2024.1`` or ``24x7:1``."""

    aliases: tuple[SymbolAlias, ...] = ()

    @field_validator("aliases")
    @classmethod
    def _check_alias_overlap(cls, aliases: tuple[SymbolAlias, ...]) -> tuple[SymbolAlias, ...]:
        by_source: dict[str, list[SymbolAlias]] = {}
        for alias in aliases:
            by_source.setdefault(alias.source, []).append(alias)
        for source, group in by_source.items():
            ordered = sorted(group, key=lambda a: a.effective_from)
            for earlier, later in pairwise(ordered):
                end = earlier.effective_to
                if end is None or end > later.effective_from:
                    raise ValueError(
                        f"overlapping symbol aliases for source {source!r}: "
                        f"{earlier.symbol!r} and {later.symbol!r} are both valid at "
                        f"{later.effective_from}; a symbol must resolve uniquely at every instant"
                    )
        return aliases

    def symbol_at(self, moment: _dt.datetime, *, source: str) -> Symbol | None:
        """Resolve the provider symbol in use at ``moment``, or None if unmapped."""
        for alias in self.aliases:
            if alias.source == source and alias.covers(moment):
                return alias.symbol
        return None


class Bar(FrozenModel):
    """One completed OHLCV interval, with an explicit availability time.

    Timing invariant: ``bar_start < bar_end <= available_at``. A bar describes a closed
    interval of market activity, so it cannot be known before that interval ends; the gap
    between ``bar_end`` and ``available_at`` is the provider's publication latency.

    ``ingested_at`` records when *our copy* arrived. It is lineage only and deliberately
    has no bearing on historical availability -- backfilling five years of data today
    must not make all of it available as of today.
    """

    instrument_id: InstrumentId
    bar_size: BarSize
    bar_start: UtcDatetime
    bar_end: UtcDatetime
    available_at: UtcDatetime

    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0)
    """Share volume for equities, base-asset volume for crypto; unit stated in the manifest."""

    vwap: float | None = None
    trade_count: int | None = Field(default=None, ge=0)

    source: str = Field(min_length=1)
    source_key: str | None = None
    """Provider record identity, where the provider offers one."""

    revision: int = Field(default=0, ge=0)
    """Source revision. A provider correction increments this and yields a new dataset."""

    ingested_at: UtcDatetime

    @model_validator(mode="after")
    def _check_timing(self) -> Self:
        if self.bar_start >= self.bar_end:
            raise ValueError(
                f"bar_start must precede bar_end, got {self.bar_start} >= {self.bar_end}; "
                "bar intervals are half-open [bar_start, bar_end) and must be non-empty"
            )
        if self.available_at < self.bar_end:
            raise ValueError(
                f"available_at {self.available_at} precedes bar_end {self.bar_end}; "
                "a completed bar cannot be available before the interval it summarises has ended"
            )
        expected = parse_duration(self.bar_size)
        actual = self.bar_end - self.bar_start
        if actual != expected:
            raise ValueError(
                f"bar interval {actual} does not match declared bar_size {self.bar_size!r} "
                f"({expected}); the label and the interval must agree"
            )
        return self

    @model_validator(mode="after")
    def _check_ohlc(self) -> Self:
        prices = {"open": self.open, "high": self.high, "low": self.low, "close": self.close}
        for name, price in prices.items():
            if not (price == price and abs(price) != float("inf")):
                raise ValueError(f"{name} price is not finite: {price!r}")
            if price <= 0:
                raise ValueError(
                    f"{name} price must be positive, got {price!r}; non-positive prices "
                    "indicate a source or normalization fault rather than a market event"
                )
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError(
                f"OHLC relationships violated: low={self.low} high={self.high} "
                f"open={self.open} close={self.close}; low must not exceed and high must "
                "not fall below the open and close"
            )
        if self.low > self.high:
            raise ValueError(f"low {self.low} exceeds high {self.high}")
        if self.vwap is not None and not (self.low <= self.vwap <= self.high):
            raise ValueError(
                f"vwap {self.vwap} falls outside the bar range [{self.low}, {self.high}]"
            )
        return self

    @property
    def key(self) -> tuple[InstrumentId, BarSize, _dt.datetime]:
        """Natural key within a dataset version."""
        return (self.instrument_id, self.bar_size, self.bar_start)

    @property
    def publication_latency(self) -> _dt.timedelta:
        """How long after the interval closed this bar became usable."""
        return self.available_at - self.bar_end
