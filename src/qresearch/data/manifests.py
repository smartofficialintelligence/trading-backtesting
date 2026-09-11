"""Dataset policies, validation reports, and the content-addressed dataset manifest.

Implements ARCHITECTURE.md sec. 4 "Dataset manifests". Two identities matter and are
deliberately kept apart:

``dataset_id``
    Hash of the *logical* inputs: source identity, normalization policy, and the version
    of the normalization code. Re-ingesting the same source under the same policy resolves
    to the same id, which is what makes a run's dataset reference meaningful. Output file
    bytes are excluded because Parquet writer metadata varies between library versions.

``content_digest``
    Hash over the canonical row content actually produced. This verifies that the files on
    disk still match what was validated, and catches a non-deterministic normalization
    step (which would show up as the same ``dataset_id`` with a different digest).
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.data.contracts import AssetClass, BarSize, Instrument, PriceAdjustment
from qresearch.ids import DatasetId, InstrumentId, content_hash
from qresearch.time import UtcDatetime

SCHEMA_VERSION = 1
"""Canonical normalized-bar schema version. Bumped only by a breaking column change."""


class TimestampLabel(StrEnum):
    """Which edge of the interval a source's bar timestamp denotes.

    The single most common cause of a one-bar look-ahead: a provider labelling bars by
    ``bar_start`` while the reader assumes the label means "data through this time".
    Adapters must declare this; it is never inferred.
    """

    BAR_START = "bar_start"
    BAR_END = "bar_end"


class MissingBarPolicy(StrEnum):
    """What normalization does with an interval that has no source data."""

    OMIT = "omit"
    """Leave the gap absent. The default: an absent bar is honest about no data."""

    ERROR = "error"
    """Fail ingestion. For sources contractually expected to be complete."""


class DuplicatePolicy(StrEnum):
    """What normalization does with two source rows for the same natural key."""

    ERROR = "error"
    KEEP_HIGHEST_REVISION = "keep_highest_revision"
    KEEP_FIRST = "keep_first"


class RevisionPolicy(StrEnum):
    """How provider corrections are represented."""

    AS_KNOWN = "as_known"
    """Every revision retained with its own ``available_at``; a point-in-time query sees
    only what was published by then. Required for honest historical simulation."""

    LATEST_KNOWN = "latest_known"
    """Only the newest revision retained. Convenient, but backtests against it are run
    on a history that was not knowable at the time."""


class NormalizationPolicy(FrozenModel):
    """Every choice that turns source rows into canonical bars.

    Part of the dataset identity: changing any field here produces a different dataset,
    because it produces different data.
    """

    timestamp_label: TimestampLabel
    publication_latency: _dt.timedelta = Field(default=_dt.timedelta(0), ge=_dt.timedelta(0))
    """Delay added to ``bar_end`` to obtain ``available_at``. Zero means the bar is
    assumed usable the instant its interval closes -- optimistic, and stated as such."""

    price_adjustment: PriceAdjustment
    missing_bars: MissingBarPolicy = MissingBarPolicy.OMIT
    duplicates: DuplicatePolicy = DuplicatePolicy.ERROR
    revisions: RevisionPolicy = RevisionPolicy.AS_KNOWN
    volume_unit: str = Field(min_length=1)
    """Documented unit, e.g. ``shares`` or ``base_asset``."""


class PartitionRef(FrozenModel):
    """One Parquet file belonging to a dataset version."""

    path: str = Field(min_length=1)
    """Relative to the dataset root, POSIX separators, so manifests survive relocation."""

    row_count: int = Field(ge=0)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    min_bar_start: UtcDatetime
    max_bar_start: UtcDatetime
    min_available_at: UtcDatetime
    max_available_at: UtcDatetime
    instrument_ids: tuple[InstrumentId, ...]

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if self.min_bar_start > self.max_bar_start:
            raise ValueError(f"min_bar_start {self.min_bar_start} > max {self.max_bar_start}")
        if self.min_available_at > self.max_available_at:
            raise ValueError(f"min_available_at {self.min_available_at} > max")
        return self


class ValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationFinding(FrozenModel):
    """One result from a data-quality check."""

    check: str = Field(min_length=1)
    severity: ValidationSeverity
    message: str = Field(min_length=1)
    instrument_id: InstrumentId | None = None
    occurrences: int = Field(default=1, ge=1)
    sample: tuple[str, ...] = ()
    """A few offending keys, for debugging without loading the dataset."""


class ValidationReport(FrozenModel):
    """Outcome of all checks run against a normalized dataset."""

    findings: tuple[ValidationFinding, ...] = ()
    checked_at: UtcDatetime

    @property
    def errors(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.severity is ValidationSeverity.WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors


class SourceRef(FrozenModel):
    """Identity of the source data a dataset was normalized from.

    ``content_sha256`` is what makes re-ingestion detectable as identical, and a corrected
    file detectable as different, without trusting filenames or modification times.
    """

    provider: str = Field(min_length=1)
    feed: str | None = None
    uri: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: int = Field(default=0, ge=0)


class DatasetIdentity(FrozenModel):
    """The hashed subset of a manifest that defines which dataset this *is*.

    Deliberately excludes creation time, ingestion time, absolute paths, and output file
    hashes, all of which vary between two runs that produce the same dataset.
    """

    schema_version: int = Field(ge=1)
    asset_class: AssetClass
    venue: str = Field(min_length=1)
    bar_size: BarSize
    calendar_id: str = Field(min_length=1)
    instrument_ids: tuple[InstrumentId, ...]
    range_start: UtcDatetime
    range_end: UtcDatetime
    policy: NormalizationPolicy
    normalization_version: str = Field(min_length=1)
    """Version of the normalization implementation. Bump when output would change."""

    sources: tuple[SourceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.range_start >= self.range_end:
            raise ValueError(
                f"range_start {self.range_start} must precede range_end {self.range_end}; "
                "dataset ranges are half-open [start, end)"
            )
        if tuple(sorted(self.instrument_ids)) != self.instrument_ids:
            raise ValueError(
                "instrument_ids must be sorted so that dataset identity does not depend "
                "on the order instruments were supplied"
            )
        if len(set(self.instrument_ids)) != len(self.instrument_ids):
            raise ValueError("instrument_ids contains duplicates")
        return self

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId(f"ds_{content_hash(self.canonical_dict())}")


class DatasetManifest(FrozenModel):
    """Immutable lineage and quality contract for an exact set of Parquet files."""

    identity: DatasetIdentity
    instruments: tuple[Instrument, ...] = ()
    """Full definitions (increments, calendar, aliases) for every id in the identity.
    Not part of the identity hash -- an alias correction is not a data change -- but
    required by the simulator, which needs quantity increments and the calendar."""

    partitions: tuple[PartitionRef, ...]
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Hash over canonical row content, independent of Parquet writer details."""

    row_count: int = Field(ge=0)
    validation: ValidationReport
    created_at: UtcDatetime
    created_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_totals(self) -> Self:
        partition_rows = sum(p.row_count for p in self.partitions)
        if partition_rows != self.row_count:
            raise ValueError(
                f"row_count {self.row_count} disagrees with the sum of partition row "
                f"counts {partition_rows}"
            )
        paths = [p.path for p in self.partitions]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate partition paths in manifest")
        if self.instruments:
            defined = {i.instrument_id for i in self.instruments}
            missing = sorted(set(self.identity.instrument_ids) - defined)
            if missing:
                raise ValueError(f"manifest lacks Instrument definitions for {missing}")
        return self

    def instrument(self, instrument_id: str) -> Instrument:
        for candidate in self.instruments:
            if candidate.instrument_id == instrument_id:
                return candidate
        raise KeyError(
            f"dataset {self.dataset_id} carries no Instrument definition for "
            f"{instrument_id!r}; re-ingest with instrument definitions"
        )

    @property
    def dataset_id(self) -> DatasetId:
        return self.identity.dataset_id

    @property
    def instrument_ids(self) -> tuple[InstrumentId, ...]:
        return self.identity.instrument_ids
