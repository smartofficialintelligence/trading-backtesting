"""Point-in-time query contracts.

The availability cutoff is expressed as a *required* field rather than an optional
argument. ARCHITECTURE.md sec. 9 lists "unsafe joins" and "cross-sectional asynchrony"
among the highest-risk leakage modes, and both begin the same way: a read that forgot to
filter on ``available_at``. Making ``as_of`` mandatory means forgetting it is a
construction error, not a silently over-permissive query.

Reading a dataset without the filter is legitimate for building manifests, computing
checksums, and plotting completed research. That path exists, but it is a separately
named method (:meth:`~qresearch.data.catalog.DatasetCatalog.scan_all_unfiltered`) so it
can never be reached by accident and is trivial to grep for in review.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import Self

import polars as pl
from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.data.contracts import BarSize
from qresearch.ids import DatasetId, InstrumentId
from qresearch.time import UtcDatetime


class BarQuery(FrozenModel):
    """A point-in-time request for canonical bars.

    Semantics, all half-open and all UTC:

    * ``available_at <= as_of`` -- the availability cutoff. Required.
    * ``start <= bar_start < end`` -- the interval of market activity of interest.

    Note that ``start``/``end`` filter on ``bar_start`` while ``as_of`` filters on
    ``available_at``. They are different axes: a bar whose interval falls inside the
    window can still be invisible because it had not been published by ``as_of``.
    """

    dataset_id: DatasetId
    as_of: UtcDatetime
    """Decision instant. No row with ``available_at > as_of`` may be returned."""

    instrument_ids: tuple[InstrumentId, ...] | None = None
    """None means every instrument in the dataset."""

    bar_size: BarSize | None = None
    start: UtcDatetime | None = None
    end: UtcDatetime | None = None
    columns: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError(
                f"start {self.start} must precede end {self.end}; bar windows are "
                "half-open [start, end)"
            )
        if self.instrument_ids is not None:
            if not self.instrument_ids:
                raise ValueError(
                    "instrument_ids is an empty tuple; pass None to mean 'all instruments' "
                    "rather than an empty selection"
                )
            if len(set(self.instrument_ids)) != len(self.instrument_ids):
                raise ValueError("instrument_ids contains duplicates")
        return self


class UnfilteredScan(FrozenModel):
    """An explicitly non-point-in-time read.

    Every construction site must state why the availability filter is being skipped. The
    reason is recorded rather than checked -- its purpose is to make an unsafe read
    visible in code review and in logs, not to be machine-validated.
    """

    dataset_id: DatasetId
    reason: str = Field(min_length=8)
    instrument_ids: tuple[InstrumentId, ...] | None = None
    bar_size: BarSize | None = None
    columns: tuple[str, ...] | None = None


class AvailabilityViolation(AssertionError):
    """Raised when a read would return data that was not yet available.

    An ``AssertionError`` because reaching this means an invariant of the storage layer
    failed, not that a caller passed bad input.
    """


# -- the as-of join --------------------------------------------------------------------


def asof_join(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    as_of_col: str,
    by: Sequence[str] = ("instrument_id",),
    right_prefix: str = "",
    max_staleness: _dt.timedelta | None = None,
) -> pl.DataFrame:
    """Attach, to each left row, the latest right observation available by that row's cutoff.

    This is the one sanctioned way to combine two point-in-time series (ARCHITECTURE.md
    sec. 9, "unsafe joins"). It differs from a bare ``join_asof`` in what it refuses:

    * The right side **must** carry an ``available_at`` column, and that is the only
      column it will match on. Joining on an event time (``bar_start``, a funding
      timestamp, a news publication time) is how a future observation gets attached to a
      past decision, so it is not offered.
    * The result is checked: every attached ``available_at`` is ``<=`` the left cutoff, or
      :class:`AvailabilityViolation` is raised. The check is cheap and guards against a
      future change to the join strategy or an unsorted-input bug.
    * Column collisions are an error, not a silent ``_right`` suffix.

    Args:
        left: rows with a cutoff column (``as_of_col``). For a bar or feature frame this
            should be its own ``available_at`` -- the instant the row itself is usable --
            not ``bar_start`` (too conservative) and never anything later.
        right: observations with ``available_at`` plus payload columns.
        as_of_col: name of the left cutoff column.
        by: key columns that must match exactly, typically the instrument.
        right_prefix: prepended to every right payload column, including the attached
            ``available_at``, so provenance stays visible (``funding_rate``,
            ``funding_available_at``).
        max_staleness: if set, a match older than this relative to the cutoff is treated
            as no match. Stale marks are a listed risk; this makes the tolerance explicit.

    Returns:
        A DataFrame with all left columns and the prefixed right payload. Left rows with
        no eligible observation carry nulls.
    """
    left_df = left.collect() if isinstance(left, pl.LazyFrame) else left
    right_df = right.collect() if isinstance(right, pl.LazyFrame) else right

    if "available_at" not in right_df.columns:
        raise ValueError(
            "the right side of an as-of join must carry an 'available_at' column; this "
            "utility only matches on availability, never on event time"
        )
    if as_of_col not in left_df.columns:
        raise ValueError(f"left side has no cutoff column {as_of_col!r}")
    missing = [k for k in by if k not in left_df.columns or k not in right_df.columns]
    if missing:
        raise ValueError(f"join keys {missing} are not present on both sides")

    attached_at = f"{right_prefix}available_at"
    payload = [c for c in right_df.columns if c not in by and c != "available_at"]
    renamed = {c: f"{right_prefix}{c}" for c in payload} | {"available_at": attached_at}
    collisions = sorted(set(renamed.values()) & set(left_df.columns))
    if collisions:
        raise ValueError(
            f"right columns {collisions} collide with left columns; pass a right_prefix"
        )

    # Sort on the time column first so Polars marks it sorted and needs no per-group
    # sortedness check; the by-keys are secondary for determinism only.
    left_sorted = left_df.sort([as_of_col, *by])
    right_sorted = right_df.rename(renamed).sort([attached_at, *by])

    joined = left_sorted.join_asof(
        right_sorted,
        left_on=as_of_col,
        right_on=attached_at,
        by=list(by),
        strategy="backward",
        tolerance=max_staleness,
        coalesce=False,
    )

    leaked = joined.filter(pl.col(attached_at) > pl.col(as_of_col))
    if not leaked.is_empty():
        raise AvailabilityViolation(
            f"as-of join attached {leaked.height} observation(s) available after the "
            f"cutoff; first: {leaked.select([*by, as_of_col, attached_at]).row(0)}"
        )
    return joined
