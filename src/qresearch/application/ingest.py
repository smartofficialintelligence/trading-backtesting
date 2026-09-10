"""Ingestion service: adapter output -> validated, manifested dataset.

Composition only. The adapter maps, :mod:`qresearch.data.validation` judges, and the
catalog stores; this module wires them in the one order that is correct and decides what
happens when the data is bad.

The default is to refuse. A dataset that fails an ERROR check is not written, because a
half-trusted dataset on disk will eventually be used by someone who did not read the
validation report.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qresearch.data.adapters.base import BarAdapter, IngestRequest
from qresearch.data.catalog import DatasetCatalog
from qresearch.data.contracts import AssetClass
from qresearch.data.manifests import (
    SCHEMA_VERSION,
    DatasetIdentity,
    DatasetManifest,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)
from qresearch.data.validation import validate_bars
from qresearch.ids import InstrumentId
from qresearch.time import as_utc_scalar, now_utc, parse_duration


class DataQualityError(ValueError):
    """Raised when ingestion is refused because validation found ERROR-level problems."""

    def __init__(self, report: ValidationReport) -> None:
        detail = "; ".join(f"{f.check}: {f.message}" for f in report.errors)
        super().__init__(f"refusing to write dataset, {len(report.errors)} error(s): {detail}")
        self.report = report


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    manifest: DatasetManifest
    report: ValidationReport
    reused_existing: bool
    """True when an identical dataset already existed and no files were rewritten."""


def ingest_bars(
    request: IngestRequest,
    *,
    catalog: DatasetCatalog,
    adapter: BarAdapter,
    asset_class: AssetClass,
    venue: str,
    calendar_id: str,
    normalization_version: str,
    created_by: str,
    expect_complete_grid: bool = False,
    allow_quality_errors: bool = False,
) -> IngestOutcome:
    """Normalize, validate, and store one batch of source bars.

    Args:
        expect_complete_grid: report every gap in the regular bar grid. Correct for 24/7
            crypto; leave off for equities until session calendars land.
        allow_quality_errors: write the dataset even if ERROR findings exist. Intended for
            deliberately-malformed test fixtures, not for research data.

    Raises:
        DataQualityError: validation found ERROR findings and ``allow_quality_errors`` is
            False.
    """
    batch = adapter.read(request)
    frame = batch.frame
    if frame.is_empty():
        raise DataQualityError(
            ValidationReport(
                findings=(
                    ValidationFinding(
                        check="empty_source",
                        severity=ValidationSeverity.ERROR,
                        message=f"source {request.uri!r} yielded no mappable rows",
                    ),
                ),
                checked_at=now_utc(),
            )
        )

    report = validate_bars(frame, policy=request.policy, expect_complete_grid=expect_complete_grid)
    if batch.dropped_unmapped_symbols:
        report = report.model_copy(
            update={
                "findings": (
                    *report.findings,
                    ValidationFinding(
                        check="unmapped_symbols",
                        severity=ValidationSeverity.WARNING,
                        message=(
                            "source rows were dropped because their symbol maps to no "
                            "requested instrument at that time"
                        ),
                        occurrences=len(batch.dropped_unmapped_symbols),
                        sample=batch.dropped_unmapped_symbols[:5],
                    ),
                )
            }
        )
    if not report.ok and not allow_quality_errors:
        raise DataQualityError(report)

    identity = DatasetIdentity(
        schema_version=SCHEMA_VERSION,
        asset_class=asset_class,
        venue=venue,
        bar_size=request.bar_size,
        calendar_id=calendar_id,
        instrument_ids=tuple(
            sorted(InstrumentId(i) for i in set(frame.get_column("instrument_id").to_list()))
        ),
        range_start=as_utc_scalar(frame.get_column("bar_start").min()),
        range_end=as_utc_scalar(frame.get_column("bar_end").max()),
        policy=request.policy,
        normalization_version=normalization_version,
        sources=(batch.source,),
    )
    existed = identity.dataset_id in catalog.list_datasets()
    manifest = catalog.write_dataset(
        identity,
        frame,
        validation=report,
        created_by=created_by,
        created_at=now_utc(),
    )
    return IngestOutcome(manifest=manifest, report=report, reused_existing=existed)


def resample_bars(frame: pl.DataFrame, *, target_bar_size: str) -> pl.DataFrame:
    """Aggregate canonical bars into a coarser canonical bar size.

    Three rules make this safe (ARCHITECTURE.md sec. 4 and the "resampling leakage" entry
    in sec. 9):

    * Windows are half-open ``[bar_start, bar_end)`` and aligned to the epoch, so a bar is
      never assembled from intervals that straddle two windows, and two datasets covering
      the same period bucket identically.
    * The output ``available_at`` is at least the **maximum** availability of its inputs.
      A five-minute bar is not usable until its slowest minute is.
    * The output ``available_at`` is also at least ``bar_end``. This is the rule that
      handles *incomplete* windows, and it matters in the direction that is easy to miss:
      a window missing its last minutes would otherwise publish early, so a consumer would
      see what looks like a finished coarse bar while inputs for that same window could
      still arrive and change it. Clamping to the window end means a coarse bar is only
      ever offered once no further input can belong to it.

    An incomplete window still produces a bar -- markets have gaps and refusing to emit
    would lose data -- but it is never visible before the window it claims to summarise
    has closed.
    """
    step = parse_duration(target_bar_size)
    sizes = set(frame.get_column("bar_size").to_list())
    if len(sizes) != 1:
        raise ValueError(f"expected exactly one source bar_size, found {sorted(sizes)}")
    source_step = parse_duration(next(iter(sizes)))
    if step <= source_step or step % source_step:
        raise ValueError(
            f"target bar size {target_bar_size!r} must be a whole multiple of the source "
            f"size {next(iter(sizes))!r} and strictly larger"
        )

    return (
        frame.sort(["instrument_id", "bar_start"])
        .group_by_dynamic(
            "bar_start",
            every=step,
            period=step,
            closed="left",
            label="left",
            group_by="instrument_id",
            start_by="window",
        )
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            # Volume-weighted where the source supplies vwap; null when it does not.
            ((pl.col("vwap") * pl.col("volume")).sum() / pl.col("volume").sum()).alias("vwap"),
            pl.col("trade_count").sum(),
            pl.col("available_at").max().alias("_inputs_available_at"),
            pl.col("source").first(),
            pl.col("revision").max(),
            pl.col("ingested_at").max(),
        )
        .with_columns(
            pl.lit(target_bar_size).alias("bar_size"),
            (pl.col("bar_start") + step).alias("bar_end"),
            pl.lit(None, dtype=pl.String).alias("source_key"),
        )
        .with_columns(
            pl.max_horizontal("_inputs_available_at", "bar_end").alias("available_at"),
        )
        .select(
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
        .sort(["instrument_id", "bar_start"])
    )
