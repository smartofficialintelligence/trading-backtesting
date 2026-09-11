"""Data-quality checks over a normalized bar frame.

The :class:`~qresearch.data.contracts.Bar` model validates one row at a time. These
checks cover the properties that only exist across rows -- duplicate keys, gaps in the
expected grid, availability that runs backwards -- and produce a
:class:`~qresearch.data.manifests.ValidationReport` that is stored inside the manifest,
so every dataset carries the evidence of what was checked and what was found.

Severity convention:

``ERROR``
    The data cannot be trusted for simulation. Ingestion should fail.
``WARNING``
    The data is usable but a result computed from it carries a caveat that must reach the
    run report (ARCHITECTURE.md sec. 8 requires data-coverage warnings alongside metrics).
``INFO``
    Descriptive, for the record.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import Final

import polars as pl

from qresearch.data.calendars import TradingCalendar, attach_sessions
from qresearch.data.manifests import (
    MissingBarPolicy,
    NormalizationPolicy,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)
from qresearch.ids import InstrumentId
from qresearch.time import as_utc_scalar, now_utc, parse_duration

MAX_SAMPLES: Final = 5
"""Offending keys retained per finding. Enough to debug, small enough to store."""

NATURAL_KEY: Final = ("instrument_id", "bar_size", "bar_start", "revision")


def _samples(frame: pl.DataFrame, columns: Sequence[str]) -> tuple[str, ...]:
    head = frame.select(columns).head(MAX_SAMPLES)
    return tuple(" ".join(str(v) for v in row) for row in head.iter_rows())


def _finding(
    check: str,
    severity: ValidationSeverity,
    message: str,
    offenders: pl.DataFrame,
    *,
    columns: Sequence[str] = NATURAL_KEY,
    instrument_id: InstrumentId | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        check=check,
        severity=severity,
        message=message,
        instrument_id=instrument_id,
        occurrences=offenders.height,
        sample=_samples(offenders, columns),
    )


def validate_bars(
    frame: pl.DataFrame,
    *,
    policy: NormalizationPolicy,
    expect_complete_grid: bool = False,
    calendar: TradingCalendar | None = None,
) -> ValidationReport:
    """Run every cross-row check against a normalized bar frame.

    Args:
        frame: bars in the canonical schema.
        policy: the normalization policy the frame was produced under. Determines whether
            gaps are an error or a warning, and what publication latency was promised.
        expect_complete_grid: when True, report a missing interval for every gap in the
            regular bar grid between an instrument's first and last observed bar. Correct
            for 24/7 crypto; for equities it produces a finding per overnight break, so
            it defaults off.
        calendar: when given, bars whose ``bar_start`` falls outside every session are
            reported (``outside_session``). Extended-hours prints in an equity dataset
            are a common way to get a "fill" at a time the venue was closed.
    """
    findings: list[ValidationFinding] = []
    if frame.is_empty():
        return ValidationReport(
            findings=(
                ValidationFinding(
                    check="empty_dataset",
                    severity=ValidationSeverity.ERROR,
                    message="the normalized frame contains no rows",
                ),
            ),
            checked_at=now_utc(),
        )

    findings.extend(_check_duplicates(frame))
    findings.extend(_check_timing(frame))
    findings.extend(_check_prices(frame))
    findings.extend(_check_availability(frame, policy=policy))
    if expect_complete_grid:
        findings.extend(_check_grid(frame, policy=policy))
    if calendar is not None:
        findings.extend(_check_sessions(frame, calendar))
    findings.append(
        ValidationFinding(
            check="row_count",
            severity=ValidationSeverity.INFO,
            message=(
                f"{frame.height} rows across "
                f"{frame.get_column('instrument_id').n_unique()} instruments"
            ),
        )
    )
    return ValidationReport(findings=tuple(findings), checked_at=now_utc())


def _check_duplicates(frame: pl.DataFrame) -> list[ValidationFinding]:
    duplicated = frame.filter(frame.select(NATURAL_KEY).is_duplicated())
    if duplicated.is_empty():
        return []
    return [
        _finding(
            "duplicate_natural_key",
            ValidationSeverity.ERROR,
            "two or more rows share (instrument_id, bar_size, bar_start, revision); the "
            "natural key must be unique within a dataset version",
            duplicated,
        )
    ]


def _check_timing(frame: pl.DataFrame) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []

    early = frame.filter(pl.col("available_at") < pl.col("bar_end"))
    if not early.is_empty():
        findings.append(
            _finding(
                "available_before_bar_end",
                ValidationSeverity.ERROR,
                "a bar is marked available before the interval it summarises has closed; "
                "this is a direct look-ahead and the dataset must not be used",
                early,
            )
        )

    inverted = frame.filter(pl.col("bar_start") >= pl.col("bar_end"))
    if not inverted.is_empty():
        findings.append(
            _finding(
                "empty_bar_interval",
                ValidationSeverity.ERROR,
                "bar_start is not strictly before bar_end",
                inverted,
            )
        )

    for bar_size in sorted(set(frame.get_column("bar_size").to_list())):
        expected = parse_duration(bar_size)
        mismatched = frame.filter(
            (pl.col("bar_size") == bar_size)
            & (
                (pl.col("bar_end") - pl.col("bar_start"))
                != pl.duration(microseconds=expected // _dt.timedelta(microseconds=1))
            )
        )
        if not mismatched.is_empty():
            findings.append(
                _finding(
                    "bar_size_mismatch",
                    ValidationSeverity.ERROR,
                    f"bars labelled {bar_size!r} do not span {expected}; the label and the "
                    "interval must agree or every downstream resample is wrong",
                    mismatched,
                )
            )
    return findings


def _check_prices(frame: pl.DataFrame) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []

    non_positive = frame.filter(pl.min_horizontal("open", "high", "low", "close") <= 0)
    if not non_positive.is_empty():
        findings.append(
            _finding(
                "non_positive_price",
                ValidationSeverity.ERROR,
                "a price is zero or negative, which indicates a source or normalization "
                "fault rather than a market event",
                non_positive,
            )
        )

    broken = frame.filter(
        (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.col("high"))
    )
    if not broken.is_empty():
        findings.append(
            _finding(
                "ohlc_relationship",
                ValidationSeverity.ERROR,
                "low exceeds, or high falls below, the open/close range",
                broken,
            )
        )

    negative_volume = frame.filter(pl.col("volume") < 0)
    if not negative_volume.is_empty():
        findings.append(
            _finding(
                "negative_volume", ValidationSeverity.ERROR, "volume is negative", negative_volume
            )
        )

    zero_volume = frame.filter(pl.col("volume") == 0)
    if not zero_volume.is_empty():
        findings.append(
            _finding(
                "zero_volume_bar",
                ValidationSeverity.WARNING,
                "bars report zero volume; any execution model sizing against traded volume "
                "must treat these as untradable rather than infinitely liquid",
                zero_volume,
            )
        )
    return findings


def _check_availability(
    frame: pl.DataFrame, *, policy: NormalizationPolicy
) -> list[ValidationFinding]:
    """Availability must be non-decreasing in bar_start, and match the declared latency."""
    findings: list[ValidationFinding] = []

    ordered = frame.sort(["instrument_id", "bar_start"])
    out_of_order = ordered.filter(
        pl.col("available_at") < pl.col("available_at").shift(1).over("instrument_id")
    )
    if not out_of_order.is_empty():
        findings.append(
            _finding(
                "availability_out_of_order",
                ValidationSeverity.WARNING,
                "a later bar became available before an earlier one for the same "
                "instrument; possible, but it means a strategy can see bar N+1 without "
                "bar N and every rolling feature must tolerate that",
                out_of_order,
            )
        )

    declared = policy.publication_latency
    unexpected = frame.filter(
        (pl.col("available_at") - pl.col("bar_end"))
        != pl.duration(microseconds=declared // _dt.timedelta(microseconds=1))
    )
    if not unexpected.is_empty():
        findings.append(
            _finding(
                "publication_latency_mismatch",
                ValidationSeverity.WARNING,
                f"publication latency differs from the declared {declared}; the manifest "
                "policy should describe what the data actually does",
                unexpected,
            )
        )

    if declared == _dt.timedelta(0):
        findings.append(
            ValidationFinding(
                check="zero_publication_latency",
                severity=ValidationSeverity.WARNING,
                message=(
                    "the policy declares zero publication latency, so bars are assumed "
                    "usable the instant their interval closes; no real feed does this and "
                    "results carry an optimistic timing assumption"
                ),
            )
        )
    return findings


def _check_sessions(frame: pl.DataFrame, calendar: TradingCalendar) -> list[ValidationFinding]:
    outside = attach_sessions(frame, calendar).filter(pl.col("session_open").is_null())
    if outside.is_empty():
        return []
    return [
        _finding(
            "outside_session",
            ValidationSeverity.WARNING,
            f"bars fall outside every {calendar.calendar_id} session (pre/post-market, "
            "weekend, or holiday); the simulator will treat their opens as tradeable",
            outside,
        )
    ]


def _check_grid(frame: pl.DataFrame, *, policy: NormalizationPolicy) -> list[ValidationFinding]:
    """Report intervals absent from the regular grid between first and last bar."""
    severity = (
        ValidationSeverity.ERROR
        if policy.missing_bars is MissingBarPolicy.ERROR
        else ValidationSeverity.WARNING
    )
    findings: list[ValidationFinding] = []
    for bar_size in sorted(set(frame.get_column("bar_size").to_list())):
        step = parse_duration(bar_size)
        subset = frame.filter(pl.col("bar_size") == bar_size)
        for instrument_id in sorted(set(subset.get_column("instrument_id").to_list())):
            series = (
                subset.filter(pl.col("instrument_id") == instrument_id)
                .get_column("bar_start")
                .unique()
                .sort()
            )
            if series.len() < 2:
                continue
            expected = pl.datetime_range(
                as_utc_scalar(series.min()),
                as_utc_scalar(series.max()),
                interval=step,
                time_zone="UTC",
                eager=True,
            )
            missing = expected.filter(~expected.is_in(series.implode()))
            missing_count = missing.len()
            if missing_count:
                findings.append(
                    ValidationFinding(
                        check="missing_bars",
                        severity=severity,
                        message=(
                            f"{missing_count} {bar_size} interval(s) between the first and "
                            f"last observed bar have no data; they are omitted rather than "
                            "filled, so features must handle the discontinuity"
                        ),
                        instrument_id=InstrumentId(instrument_id),
                        occurrences=missing_count,
                        sample=tuple(str(v) for v in missing.head(MAX_SAMPLES).to_list()),
                    )
                )
    return findings
