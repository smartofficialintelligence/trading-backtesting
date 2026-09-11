"""Local Parquet storage and the point-in-time bar catalog.

Parquet is the source of truth (ARCHITECTURE.md sec. 4); DuckDB and any index built on
top are rebuildable views. This module owns the canonical on-disk schema, the layout, and
the only two ways to read bars: a point-in-time scan, and an explicitly-labelled
unfiltered scan.

Layout::

    <root>/normalized/schema_version=1/asset_class=<c>/bar_size=<s>/dataset=<id>/date=<YYYY-MM-DD>/part-0000.parquet
    <root>/manifests/<dataset_id>.json

``dataset=<id>`` is what makes datasets immutable. Without it, a corrected version of the
same instruments over the same days would write to the same files as the original and
silently invalidate every manifest that referenced them -- so a run pinned to the old
dataset id would quietly start reading the new data. ``schema_version``, ``asset_class``,
and ``bar_size`` sit above it (they are fixed within a dataset) so that a cross-dataset
DuckDB view can still prune on them.

``date`` is the UTC date of ``bar_start``. ARCHITECTURE.md sketches an additional
per-symbol partition level; that is deliberately omitted here in favour of the same
section's instruction that files be "compacted to useful analytical sizes rather than
producing one tiny file per symbol-minute". One file per UTC day holds every instrument,
rows sorted by ``(instrument_id, bar_start)``, so Parquet row-group statistics still
prune by instrument without multiplying file count by universe size.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from qresearch.data.contracts import Instrument
from qresearch.data.manifests import (
    DatasetIdentity,
    DatasetManifest,
    PartitionRef,
    ValidationReport,
)
from qresearch.data.point_in_time import (
    AvailabilityViolation,
    BarQuery,
    UnfilteredScan,
)
from qresearch.ids import DatasetId, InstrumentId, content_hash_full, file_hash_full
from qresearch.time import as_utc_scalar, ensure_utc

_TS: Final = pa.timestamp("us", tz="UTC")

BAR_SCHEMA: Final = pa.schema(
    [
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("bar_size", pa.string(), nullable=False),
        pa.field("bar_start", _TS, nullable=False),
        pa.field("bar_end", _TS, nullable=False),
        pa.field("available_at", _TS, nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("vwap", pa.float64(), nullable=True),
        pa.field("trade_count", pa.int64(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_key", pa.string(), nullable=True),
        pa.field("revision", pa.int32(), nullable=False),
        pa.field("ingested_at", _TS, nullable=False),
    ]
)
"""Canonical normalized-bar schema. Microsecond UTC timestamps throughout."""

SORT_KEY: Final = ("instrument_id", "bar_start", "revision")
"""Canonical row order. Fixed so that content digests and scans are order-independent."""

_DIGEST_COLUMNS: Final = tuple(
    f.name for f in BAR_SCHEMA if f.name not in {"ingested_at", "source_key"}
)
"""Columns entering the content digest.

``ingested_at`` is wall-clock lineage and differs between two identical ingestions;
``source_key`` is provider bookkeeping. Excluding both lets the digest answer "is this the
same market data?" rather than "was this written by the same process run?".
"""


class DatasetNotFoundError(LookupError):
    """Raised when a dataset id has no manifest under the catalog root."""


def _partition_dir(root: Path, identity: DatasetIdentity, day: _dt.date) -> Path:
    return (
        root
        / "normalized"
        / f"schema_version={identity.schema_version}"
        / f"asset_class={identity.asset_class.value}"
        / f"bar_size={identity.bar_size}"
        / f"dataset={identity.dataset_id}"
        / f"date={day.isoformat()}"
    )


def bars_to_arrow(frame: pl.DataFrame) -> pa.Table:
    """Cast a bar frame to the canonical Arrow schema, in canonical row order."""
    missing = [f.name for f in BAR_SCHEMA if f.name not in frame.columns]
    if missing:
        raise ValueError(f"bar frame is missing required columns: {missing}")
    ordered = frame.select([f.name for f in BAR_SCHEMA]).sort(SORT_KEY)
    return ordered.to_arrow().cast(BAR_SCHEMA)


def compute_content_digest(frame: pl.DataFrame) -> str:
    """Hash the canonical market content of a bar frame.

    Independent of Parquet writer version, compression, and row-group layout, so an
    identical re-ingestion produces an identical digest even when the files differ byte
    for byte (DEVELOPMENT_PLAN.md sec. 7).
    """
    ordered = frame.select(list(_DIGEST_COLUMNS)).sort(
        [c for c in SORT_KEY if c in _DIGEST_COLUMNS]
    )
    rows = ordered.to_dicts()
    return content_hash_full(rows)


class DatasetCatalog:
    """Reads and writes normalized bar datasets under one local root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- manifests ---------------------------------------------------------------

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifests"

    def manifest_path(self, dataset_id: DatasetId) -> Path:
        return self.manifest_dir / f"{dataset_id}.json"

    def resolve(self, dataset_id: DatasetId) -> DatasetManifest:
        """Load the manifest for ``dataset_id``."""
        path = self.manifest_path(dataset_id)
        if not path.exists():
            raise DatasetNotFoundError(
                f"no manifest for dataset {dataset_id!r} under {self.manifest_dir}"
            )
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def list_datasets(self) -> tuple[DatasetId, ...]:
        if not self.manifest_dir.exists():
            return ()
        return tuple(sorted(DatasetId(p.stem) for p in self.manifest_dir.glob("*.json")))

    def _write_manifest(self, manifest: DatasetManifest) -> None:
        """Persist a manifest, refusing to overwrite a differing one.

        A dataset id is a hash of its logical inputs. Two manifests sharing an id but
        differing in content mean normalization is not deterministic -- that must surface
        loudly rather than have one silently replace the other.
        """
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifest_path(manifest.dataset_id)
        payload = json.dumps(json.loads(manifest.model_dump_json()), sort_keys=True, indent=2)
        if path.exists():
            existing = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
            if existing.content_digest != manifest.content_digest:
                raise ValueError(
                    f"dataset {manifest.dataset_id} already exists with content digest "
                    f"{existing.content_digest[:16]}... but the new manifest has "
                    f"{manifest.content_digest[:16]}...; identical logical inputs produced "
                    "different data, which means normalization is not deterministic"
                )
            return
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    # -- writing -----------------------------------------------------------------

    def write_dataset(
        self,
        identity: DatasetIdentity,
        frame: pl.DataFrame,
        *,
        validation: ValidationReport,
        created_by: str,
        created_at: _dt.datetime,
        instruments: Sequence[Instrument] = (),
    ) -> DatasetManifest:
        """Write ``frame`` as a new dataset version and return its manifest.

        Files are written per UTC day of ``bar_start``. Data files are never overwritten:
        if the dataset already exists with the same content digest the write is a no-op,
        and if the digest differs it is an error.
        """
        table = bars_to_arrow(frame)
        digest = compute_content_digest(frame)
        existing = self._existing_manifest(identity.dataset_id)
        if existing is not None and existing.content_digest == digest:
            return existing

        ordered = pl.from_arrow(table)
        assert isinstance(ordered, pl.DataFrame)
        partitions: list[PartitionRef] = []
        day_column = ordered.get_column("bar_start").dt.date()
        for day in sorted(set(day_column.to_list())):
            chunk = ordered.filter(day_column == day)
            directory = _partition_dir(self.root, identity, day)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "part-0000.parquet"
            pq.write_table(
                chunk.to_arrow().cast(BAR_SCHEMA),
                path,
                compression="zstd",
                # Fixed writer settings keep output stable run to run; the content digest
                # is the authority, but stable files make diffs and caching behave.
                write_statistics=True,
                use_dictionary=["instrument_id", "bar_size", "source"],
            )
            partitions.append(
                PartitionRef(
                    path=path.relative_to(self.root).as_posix(),
                    row_count=chunk.height,
                    byte_size=path.stat().st_size,
                    sha256=file_hash_full(path),
                    min_bar_start=as_utc_scalar(chunk.get_column("bar_start").min()),
                    max_bar_start=as_utc_scalar(chunk.get_column("bar_start").max()),
                    min_available_at=as_utc_scalar(chunk.get_column("available_at").min()),
                    max_available_at=as_utc_scalar(chunk.get_column("available_at").max()),
                    instrument_ids=tuple(
                        InstrumentId(i)
                        for i in sorted(set(chunk.get_column("instrument_id").to_list()))
                    ),
                )
            )

        manifest = DatasetManifest(
            identity=identity,
            instruments=tuple(sorted(instruments, key=lambda i: i.instrument_id)),
            partitions=tuple(partitions),
            content_digest=digest,
            row_count=ordered.height,
            validation=validation,
            created_at=ensure_utc(created_at),
            created_by=created_by,
        )
        self._write_manifest(manifest)
        return manifest

    def _existing_manifest(self, dataset_id: DatasetId) -> DatasetManifest | None:
        try:
            return self.resolve(dataset_id)
        except DatasetNotFoundError:
            return None

    # -- reading -----------------------------------------------------------------

    def scan_bars(self, query: BarQuery) -> pl.LazyFrame:
        """Point-in-time scan: no returned row has ``available_at > query.as_of``.

        Partition pruning uses each partition's recorded ``min_available_at``, so a bar
        published late is still excluded correctly even though it lives in an early
        ``date=`` directory.
        """
        manifest = self.resolve(query.dataset_id)
        paths = self._select_partitions(
            manifest,
            as_of=query.as_of,
            instrument_ids=query.instrument_ids,
            bar_size=query.bar_size,
            start=query.start,
            end=query.end,
        )
        lazy = self._scan_paths(paths)
        lazy = lazy.filter(pl.col("available_at") <= query.as_of)
        if query.instrument_ids is not None:
            lazy = lazy.filter(pl.col("instrument_id").is_in(list(query.instrument_ids)))
        if query.bar_size is not None:
            lazy = lazy.filter(pl.col("bar_size") == query.bar_size)
        if query.start is not None:
            lazy = lazy.filter(pl.col("bar_start") >= query.start)
        if query.end is not None:
            lazy = lazy.filter(pl.col("bar_start") < query.end)
        lazy = lazy.sort(SORT_KEY)
        if query.columns is not None:
            # Sort first: the canonical sort key may itself be projected away.
            lazy = self._select_columns(lazy, query.columns)
        return lazy

    def scan_all_unfiltered(self, scan: UnfilteredScan) -> pl.LazyFrame:
        """Read a dataset **without** the availability filter.

        Only for manifest construction, integrity checks, and post-hoc analysis of
        completed research. Never reachable from strategy, feature, or simulation code.
        """
        manifest = self.resolve(scan.dataset_id)
        lazy = self._scan_paths([self.root / p.path for p in manifest.partitions])
        if scan.instrument_ids is not None:
            lazy = lazy.filter(pl.col("instrument_id").is_in(list(scan.instrument_ids)))
        if scan.bar_size is not None:
            lazy = lazy.filter(pl.col("bar_size") == scan.bar_size)
        lazy = lazy.sort(SORT_KEY)
        if scan.columns is not None:
            lazy = self._select_columns(lazy, scan.columns)
        return lazy

    @staticmethod
    def _select_columns(lazy: pl.LazyFrame, columns: Sequence[str]) -> pl.LazyFrame:
        unknown = [c for c in columns if c not in BAR_SCHEMA.names]
        if unknown:
            raise ValueError(f"unknown bar columns requested: {unknown}")
        return lazy.select(list(columns))

    def _scan_paths(self, paths: Sequence[Path]) -> pl.LazyFrame:
        if not paths:
            return pl.LazyFrame(schema=_polars_schema())
        return pl.scan_parquet([str(p) for p in paths])

    def _select_partitions(
        self,
        manifest: DatasetManifest,
        *,
        as_of: _dt.datetime,
        instrument_ids: Iterable[InstrumentId] | None,
        bar_size: str | None,
        start: _dt.datetime | None,
        end: _dt.datetime | None,
    ) -> list[Path]:
        """Drop partitions that provably contain no matching row."""
        if bar_size is not None and bar_size != manifest.identity.bar_size:
            return []
        wanted = set(instrument_ids) if instrument_ids is not None else None
        selected: list[Path] = []
        for partition in manifest.partitions:
            if partition.min_available_at > as_of:
                continue
            if start is not None and partition.max_bar_start < start:
                continue
            if end is not None and partition.min_bar_start >= end:
                continue
            if wanted is not None and not wanted.intersection(partition.instrument_ids):
                continue
            selected.append(self.root / partition.path)
        return selected

    # -- integrity ---------------------------------------------------------------

    def verify(self, dataset_id: DatasetId) -> None:
        """Check that files on disk still match the manifest.

        Raises:
            FileNotFoundError: a recorded partition is missing.
            AvailabilityViolation: a partition holds a row later than it declares, which
                would let point-in-time pruning skip a row it should have returned.
            ValueError: a file's bytes or the dataset's content no longer match.
        """
        manifest = self.resolve(dataset_id)
        for partition in manifest.partitions:
            path = self.root / partition.path
            if not path.exists():
                raise FileNotFoundError(f"partition {partition.path} is missing from {self.root}")
            actual = file_hash_full(path)
            if actual != partition.sha256:
                raise ValueError(
                    f"partition {partition.path} has sha256 {actual[:16]}... but the "
                    f"manifest records {partition.sha256[:16]}...; the file has changed "
                    "since it was validated"
                )
        frame = self.scan_all_unfiltered(
            UnfilteredScan(dataset_id=dataset_id, reason="manifest integrity verification")
        ).collect()
        for partition in manifest.partitions:
            chunk = frame.filter(
                pl.col("bar_start").dt.date()
                == _dt.date.fromisoformat(Path(partition.path).parent.name.removeprefix("date="))
            )
            if (
                chunk.height
                and as_utc_scalar(chunk.get_column("available_at").max())
                > partition.max_available_at
            ):
                raise AvailabilityViolation(
                    f"partition {partition.path} contains a row available after its "
                    "recorded max_available_at; point-in-time pruning would skip it"
                )
        digest = compute_content_digest(frame)
        if digest != manifest.content_digest:
            raise ValueError(
                f"dataset {dataset_id} content digest is {digest[:16]}... but the manifest "
                f"records {manifest.content_digest[:16]}..."
            )


def _polars_schema() -> dict[str, pl.DataType]:
    """Empty-frame schema matching :data:`BAR_SCHEMA`."""
    empty = pl.from_arrow(BAR_SCHEMA.empty_table())
    assert isinstance(empty, pl.DataFrame)
    return dict(empty.schema)
