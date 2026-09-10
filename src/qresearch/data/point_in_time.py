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

from typing import Self

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
