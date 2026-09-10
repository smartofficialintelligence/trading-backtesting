"""Provider timestamp-label convention.

DEVELOPMENT_PLAN.md Stage 1 requires that "provider timestamp behavior is captured in
contract tests". This is the single highest-value test in the data layer: a source that
labels bars by ``bar_end`` read as if it labelled them by ``bar_start`` produces a dataset
shifted one bar into the past, so every strategy trades on information it could not have
had -- and the equity curve looks merely good, not broken.

The two conventions must therefore produce *demonstrably different* datasets from the same
bytes, and each must place the interval where the provider says it is.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
from tests.conftest import make_request, run_ingest

from qresearch.data.adapters.base import derive_bar_bounds
from qresearch.data.manifests import TimestampLabel
from qresearch.data.point_in_time import UnfilteredScan

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)


def _frame(timestamps: list[dt.datetime]) -> pl.DataFrame:
    return pl.DataFrame({"ts": timestamps}, schema={"ts": pl.Datetime("us", "UTC")})


def test_bar_start_label_places_the_interval_after_the_timestamp() -> None:
    out = derive_bar_bounds(
        _frame([T0]), bar_size="1m", label=TimestampLabel.BAR_START, timestamp_column="ts"
    )
    assert out.item(0, "bar_start") == T0
    assert out.item(0, "bar_end") == T0 + dt.timedelta(minutes=1)


def test_bar_end_label_places_the_interval_before_the_timestamp() -> None:
    out = derive_bar_bounds(
        _frame([T0]), bar_size="1m", label=TimestampLabel.BAR_END, timestamp_column="ts"
    )
    assert out.item(0, "bar_start") == T0 - dt.timedelta(minutes=1)
    assert out.item(0, "bar_end") == T0


def test_the_two_conventions_differ_by_exactly_one_bar() -> None:
    starts = {
        label: derive_bar_bounds(
            _frame([T0]), bar_size="5m", label=label, timestamp_column="ts"
        ).item(0, "bar_start")
        for label in TimestampLabel
    }
    assert starts[TimestampLabel.BAR_START] - starts[TimestampLabel.BAR_END] == dt.timedelta(
        minutes=5
    )


def test_availability_is_never_earlier_than_the_interval_end_under_either_label() -> None:
    """Whichever edge the source labels, a bar cannot precede its own data."""
    for label in TimestampLabel:
        out = derive_bar_bounds(_frame([T0]), bar_size="1m", label=label, timestamp_column="ts")
        assert out.item(0, "bar_start") < out.item(0, "bar_end")


@pytest.mark.parametrize("label", list(TimestampLabel))
def test_reading_a_source_under_each_label_yields_different_datasets(
    label: TimestampLabel, catalog, source_csv: Path, spec, policy
) -> None:
    """The same bytes under a different declared label must be a different dataset id.

    If both conventions collapsed to one id, a mislabelled ingestion could silently
    overwrite or alias a correct one.
    """
    outcome = run_ingest(
        catalog,
        make_request(source_csv, spec, policy.model_copy(update={"timestamp_label": label})),
        expect_complete_grid=False,
    )
    frame = catalog.scan_all_unfiltered(
        UnfilteredScan(dataset_id=outcome.manifest.dataset_id, reason="label convention check")
    ).collect()
    first = frame.filter(pl.col("instrument_id") == "CRYPTO:BTCUSD").row(0, named=True)
    expected_start = T0 if label is TimestampLabel.BAR_START else T0 - dt.timedelta(minutes=1)
    assert first["bar_start"] == expected_start
    assert first["bar_end"] == expected_start + dt.timedelta(minutes=1)


def test_a_mislabelled_read_is_visibly_a_different_dataset(
    catalog, source_csv: Path, spec, policy
) -> None:
    ids = {
        label: run_ingest(
            catalog,
            make_request(source_csv, spec, policy.model_copy(update={"timestamp_label": label})),
            expect_complete_grid=False,
        ).manifest.dataset_id
        for label in TimestampLabel
    }
    assert len(set(ids.values())) == 2


def test_naive_source_timestamps_are_refused(catalog, tmp_path: Path, spec, policy) -> None:
    """An exchange-local export without an offset must fail loudly, not be assumed UTC."""
    source = tmp_path / "naive.csv"
    source.write_text(
        "timestamp,symbol,open,high,low,close,volume,trades,available_at,revision\n"
        "2024-03-04T00:00:00,BTC-USD,100,101,99,100.5,10,3,2024-03-04T00:01:02Z,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no UTC offset"):
        run_ingest(catalog, make_request(source, spec, policy))
