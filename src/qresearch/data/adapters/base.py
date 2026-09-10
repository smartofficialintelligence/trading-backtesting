"""Adapter contract for turning source bars into canonical bars.

The one thing every adapter must state explicitly is
:class:`~qresearch.data.manifests.TimestampLabel`: whether a source's bar timestamp means
``bar_start`` or ``bar_end``. Getting this wrong shifts the whole dataset by one bar and
produces a backtest that trades on information from the future -- and it produces
*plausible* results, which is what makes it dangerous. It is therefore a required field
with no default, and it is covered by provider contract tests.

Adapters do no quality filtering. They map, they compute availability, and they hand back
a canonical frame; judging the data is :mod:`qresearch.data.validation`'s job.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Protocol, Self

import polars as pl
from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.data.contracts import BarSize, Instrument
from qresearch.data.manifests import (
    DuplicatePolicy,
    NormalizationPolicy,
    SourceRef,
    TimestampLabel,
)
from qresearch.ids import InstrumentId
from qresearch.time import parse_duration


class ColumnMapping(FrozenModel):
    """Source column names for each canonical field."""

    timestamp: str = "timestamp"
    symbol: str = "symbol"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    vwap: str | None = None
    trade_count: str | None = None
    source_key: str | None = None
    revision: str | None = None
    available_at: str | None = None
    """Source-provided availability, when the provider publishes one.

    Preferred over the derived ``bar_end + publication_latency`` because it is the truth
    rather than an assumption. When absent, availability is derived and the manifest's
    policy records the latency that was assumed.
    """


class IngestRequest(FrozenModel):
    """Everything needed to normalize one batch of source bars."""

    uri: str = Field(min_length=1)
    bar_size: BarSize
    policy: NormalizationPolicy
    instruments: tuple[Instrument, ...] = Field(min_length=1)
    source_name: str = Field(min_length=1)
    """Value written to each bar's ``source`` column, e.g. ``binance:spot-klines``."""

    mapping: ColumnMapping = ColumnMapping()
    source_revision: int = Field(default=0, ge=0)
    feed: str | None = None

    @model_validator(mode="after")
    def _check_instruments(self) -> Self:
        ids = [i.instrument_id for i in self.instruments]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate instrument_id in request instruments")
        return self

    @property
    def instrument_by_id(self) -> dict[InstrumentId, Instrument]:
        return {i.instrument_id: i for i in self.instruments}


@dataclass(frozen=True, slots=True)
class NormalizedBatch:
    """An adapter's output: canonical bars plus the identity of what produced them.

    A dataclass rather than a pydantic model because it carries a live ``DataFrame``; it
    is an in-process handoff, not a persisted contract.
    """

    frame: pl.DataFrame
    source: SourceRef
    dropped_unmapped_symbols: tuple[str, ...] = ()
    """Source symbols with no instrument mapping, recorded rather than silently dropped."""


class BarAdapter(Protocol):
    """Reads a source of bar data and returns canonical bars."""

    def read(self, request: IngestRequest) -> NormalizedBatch: ...


def resolve_symbols(
    frame: pl.DataFrame,
    request: IngestRequest,
    *,
    symbol_column: str,
    at_column: str,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Map provider symbols to ``instrument_id`` using effective-dated aliases.

    Resolution is point-in-time: the alias valid at the bar's own timestamp wins, so a
    ticker change does not retroactively rewrite the identity of older bars.
    """
    intervals: list[dict[str, object]] = []
    for instrument in request.instruments:
        for alias in instrument.aliases:
            if alias.source != request.source_name:
                continue
            intervals.append(
                {
                    "_alias_symbol": alias.symbol,
                    "_alias_instrument_id": str(instrument.instrument_id),
                    "_alias_from": alias.effective_from,
                    "_alias_to": alias.effective_to or _dt.datetime.max.replace(tzinfo=_dt.UTC),
                }
            )
    if not intervals:
        raise ValueError(
            f"no symbol aliases for source {request.source_name!r} on any requested "
            "instrument; a provider symbol cannot be mapped to a stable instrument_id"
        )
    alias_frame = pl.DataFrame(
        intervals,
        schema={
            "_alias_symbol": pl.String,
            "_alias_instrument_id": pl.String,
            "_alias_from": pl.Datetime("us", "UTC"),
            "_alias_to": pl.Datetime("us", "UTC"),
        },
    )
    joined = frame.join(
        alias_frame, left_on=symbol_column, right_on="_alias_symbol", how="left"
    ).with_columns(
        pl.when(
            (pl.col(at_column) >= pl.col("_alias_from")) & (pl.col(at_column) < pl.col("_alias_to"))
        )
        .then(pl.col("_alias_instrument_id"))
        .otherwise(None)
        .alias("instrument_id")
    )
    unmapped = joined.filter(pl.col("instrument_id").is_null())
    dropped = tuple(sorted(set(unmapped.get_column(symbol_column).to_list())))
    resolved = joined.filter(pl.col("instrument_id").is_not_null()).drop(
        "_alias_instrument_id", "_alias_from", "_alias_to"
    )
    return resolved, dropped


def derive_bar_bounds(
    frame: pl.DataFrame, *, bar_size: BarSize, label: TimestampLabel, timestamp_column: str
) -> pl.DataFrame:
    """Turn a single source timestamp into ``bar_start`` and ``bar_end``.

    This is the one place the label convention is applied. Everything downstream reads
    both edges explicitly and never has to guess.
    """
    step = parse_duration(bar_size)
    micros = step // _dt.timedelta(microseconds=1)
    ts = pl.col(timestamp_column)
    match label:
        case TimestampLabel.BAR_START:
            start, end = ts, ts + pl.duration(microseconds=micros)
        case TimestampLabel.BAR_END:
            start, end = ts - pl.duration(microseconds=micros), ts
    return frame.with_columns(start.alias("bar_start"), end.alias("bar_end"))


def apply_duplicate_policy(frame: pl.DataFrame, policy: DuplicatePolicy) -> pl.DataFrame:
    """Resolve rows sharing ``(instrument_id, bar_size, bar_start)``."""
    key = ["instrument_id", "bar_size", "bar_start"]
    duplicated = frame.filter(frame.select(key).is_duplicated())
    if duplicated.is_empty():
        return frame
    match policy:
        case DuplicatePolicy.ERROR:
            sample = duplicated.select(key).head(5).rows()
            raise ValueError(
                f"{duplicated.height} duplicate source rows for keys such as {sample}; "
                "the policy is DuplicatePolicy.ERROR"
            )
        case DuplicatePolicy.KEEP_HIGHEST_REVISION:
            return frame.sort([*key, "revision"]).unique(
                subset=key, keep="last", maintain_order=True
            )
        case DuplicatePolicy.KEEP_FIRST:
            return frame.unique(subset=key, keep="first", maintain_order=True)
