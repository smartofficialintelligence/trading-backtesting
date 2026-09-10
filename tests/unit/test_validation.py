"""Cross-row data-quality checks."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.unit.test_resample import minute_bars

from qresearch.data.contracts import PriceAdjustment
from qresearch.data.manifests import (
    MissingBarPolicy,
    NormalizationPolicy,
    TimestampLabel,
    ValidationSeverity,
)
from qresearch.data.validation import validate_bars

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)


def policy(**overrides: object) -> NormalizationPolicy:
    base: dict[str, object] = {
        "timestamp_label": TimestampLabel.BAR_START,
        "publication_latency": dt.timedelta(seconds=2),
        "price_adjustment": PriceAdjustment.NOT_APPLICABLE,
        "volume_unit": "base_asset",
    }
    return NormalizationPolicy.model_validate(base | overrides)


def checks(frame: pl.DataFrame, *, severity: ValidationSeverity | None = None, **kwargs: object):
    report = validate_bars(frame, policy=kwargs.pop("policy", policy()), **kwargs)  # type: ignore[arg-type]
    return {f.check for f in report.findings if severity is None or f.severity is severity}


def test_clean_data_produces_no_errors() -> None:
    report = validate_bars(minute_bars(30), policy=policy())
    assert report.ok
    assert report.errors == ()


def test_an_empty_frame_is_an_error() -> None:
    report = validate_bars(minute_bars(0), policy=policy())
    assert not report.ok
    assert "empty_dataset" in {f.check for f in report.errors}


def test_a_bar_available_before_its_interval_closes_is_an_error() -> None:
    """The direct look-ahead. Must be an ERROR, never a warning."""
    frame = minute_bars(5).with_columns(
        pl.when(pl.col("bar_start") == T0)
        .then(pl.col("bar_start"))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    assert "available_before_bar_end" in checks(frame, severity=ValidationSeverity.ERROR)


def test_duplicate_natural_keys_are_an_error() -> None:
    frame = pl.concat([minute_bars(3), minute_bars(3)])
    assert "duplicate_natural_key" in checks(frame, severity=ValidationSeverity.ERROR)


def test_a_bar_size_label_that_disagrees_with_the_interval_is_an_error() -> None:
    frame = minute_bars(3).with_columns(pl.lit("5m").alias("bar_size"))
    assert "bar_size_mismatch" in checks(frame, severity=ValidationSeverity.ERROR)


def test_broken_ohlc_relationships_are_an_error() -> None:
    frame = minute_bars(3).with_columns(pl.lit(1000.0).alias("low"))
    assert "ohlc_relationship" in checks(frame, severity=ValidationSeverity.ERROR)


def test_non_positive_prices_are_an_error() -> None:
    frame = minute_bars(3).with_columns(pl.lit(0.0).alias("low"))
    assert "non_positive_price" in checks(frame, severity=ValidationSeverity.ERROR)


def test_zero_volume_is_a_warning_not_an_error() -> None:
    """A halted or untraded minute is real data; it just must not be treated as liquid."""
    frame = minute_bars(3).with_columns(pl.lit(0.0).alias("volume"))
    assert "zero_volume_bar" in checks(frame, severity=ValidationSeverity.WARNING)
    assert validate_bars(frame, policy=policy()).ok


def test_a_late_bar_makes_availability_run_backwards() -> None:
    frame = minute_bars(5, late={2: dt.timedelta(minutes=9)})
    found = checks(frame, severity=ValidationSeverity.WARNING)
    assert "availability_out_of_order" in found
    assert "publication_latency_mismatch" in found


def test_zero_declared_latency_is_flagged_as_optimistic() -> None:
    """No real feed publishes instantly; the assumption must reach the run report."""
    frame = minute_bars(3).with_columns(pl.col("bar_end").alias("available_at"))
    assert "zero_publication_latency" in checks(
        frame, policy=policy(publication_latency=dt.timedelta(0))
    )


def test_gaps_are_reported_only_when_a_complete_grid_is_expected() -> None:
    frame = minute_bars(10, skip={4, 5})
    assert "missing_bars" not in checks(frame)
    assert "missing_bars" in checks(frame, expect_complete_grid=True)


def test_gap_severity_follows_the_missing_bar_policy() -> None:
    frame = minute_bars(10, skip={4, 5})
    strict = validate_bars(
        frame, policy=policy(missing_bars=MissingBarPolicy.ERROR), expect_complete_grid=True
    )
    assert not strict.ok
    lenient = validate_bars(
        frame, policy=policy(missing_bars=MissingBarPolicy.OMIT), expect_complete_grid=True
    )
    assert lenient.ok


def test_findings_carry_bounded_samples() -> None:
    frame = minute_bars(50).with_columns(pl.lit(1000.0).alias("low"))
    finding = next(f for f in validate_bars(frame, policy=policy()).errors)
    assert finding.occurrences == 50
    assert 0 < len(finding.sample) <= 5


@pytest.mark.parametrize("expect_grid", [True, False])
def test_validation_is_deterministic(expect_grid: bool) -> None:
    frame = minute_bars(20, skip={7}, late={3: dt.timedelta(minutes=5)})
    a = validate_bars(frame, policy=policy(), expect_complete_grid=expect_grid)
    b = validate_bars(frame, policy=policy(), expect_complete_grid=expect_grid)
    assert [f.model_dump() for f in a.findings] == [f.model_dump() for f in b.findings]
