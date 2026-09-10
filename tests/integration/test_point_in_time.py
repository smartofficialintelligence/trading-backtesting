"""Point-in-time reads: DEVELOPMENT_PLAN.md Stage 1 acceptance criteria.

Every test here maps to a stated criterion:

* "Queries at time t return no row with available_at > t"
* "Re-ingesting the same source and policy yields the same logical dataset ID"
* "Corrections produce a new manifest without changing old data"
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from tests.conftest import IngestedFixture, make_request, run_ingest

from qresearch.data.catalog import DatasetCatalog
from qresearch.data.point_in_time import BarQuery, UnfilteredScan
from qresearch.data.synthetic import SyntheticSpec, write_source_csv

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)
BTC = "CRYPTO:BTCUSD"
ETH = "CRYPTO:ETHUSD"


def at(minute: int, second: int = 0) -> dt.datetime:
    return T0 + dt.timedelta(minutes=minute, seconds=second)


def scan(fixture: IngestedFixture, **kwargs: object) -> pl.DataFrame:
    return fixture.catalog.scan_bars(
        BarQuery.model_validate({"dataset_id": fixture.dataset_id} | kwargs)
    ).collect()


# -- the core invariant --------------------------------------------------------------


@pytest.mark.parametrize("minute", [0, 1, 5, 31, 37, 40, 120, 239, 300])
def test_no_row_is_ever_returned_before_it_was_available(
    ingested: IngestedFixture, minute: int
) -> None:
    frame = scan(ingested, as_of=at(minute))
    if frame.is_empty():
        return
    assert frame.get_column("available_at").max() <= at(minute)


def test_a_late_published_bar_is_invisible_until_it_publishes(
    ingested: IngestedFixture,
) -> None:
    """Minute 30 is published 7 minutes late; minutes 31+ arrive on time.

    The correct behaviour is a *hole*: the series jumps 00:29 -> 00:31 and only later
    fills in. A loader that instead returned the newest N bars, or that treated the gap as
    end-of-data, would quietly hand the strategy a bar it could not have seen.
    """
    before = scan(ingested, as_of=at(33), instrument_ids=(BTC,))
    starts = set(before.get_column("bar_start").to_list())
    assert at(30) not in starts, "a bar published at 00:38 was visible at 00:33"
    assert at(31) in starts, "an on-time later bar should still be visible"
    assert at(29) in starts

    after = scan(ingested, as_of=at(39), instrument_ids=(BTC,))
    assert at(30) in set(after.get_column("bar_start").to_list())


def test_assets_are_asynchronous_at_the_same_instant(ingested: IngestedFixture) -> None:
    """At 00:33 the two assets have different newest bars.

    ARCHITECTURE.md sec. 9 lists cross-sectional asynchrony as a leakage mode: treating
    one asset's delayed bar as though it arrived with the rest. The loader must expose the
    asymmetry rather than paper over it.
    """
    frame = scan(ingested, as_of=at(33))
    counts = dict(frame.group_by("instrument_id").len().rows())
    assert counts[BTC] != counts[ETH]


def test_availability_and_window_filters_are_independent_axes(
    ingested: IngestedFixture,
) -> None:
    """A bar inside the requested window can still be invisible for not being published."""
    frame = scan(ingested, as_of=at(33), start=at(28), end=at(33), instrument_ids=(BTC,))
    starts = set(frame.get_column("bar_start").to_list())
    # 00:30 is inside the window but published late (00:38). 00:32 is inside the window
    # too, but its interval only closes at 00:33 and it publishes at 00:33:02.
    assert starts == {at(28), at(29), at(31)}


def test_the_window_end_is_exclusive(ingested: IngestedFixture) -> None:
    frame = scan(ingested, as_of=at(300), start=at(10), end=at(13), instrument_ids=(BTC,))
    assert sorted(frame.get_column("bar_start").to_list()) == [at(10), at(11), at(12)]


def test_a_query_before_any_data_returns_an_empty_typed_frame(
    ingested: IngestedFixture,
) -> None:
    frame = scan(ingested, as_of=T0 - dt.timedelta(days=1))
    assert frame.is_empty()
    assert frame.schema["bar_start"] == pl.Datetime("us", "UTC")


def test_results_do_not_depend_on_instrument_argument_order(
    ingested: IngestedFixture,
) -> None:
    forward = scan(ingested, as_of=at(300), instrument_ids=(BTC, ETH))
    reverse = scan(ingested, as_of=at(300), instrument_ids=(ETH, BTC))
    assert forward.equals(reverse)


def test_column_projection_preserves_point_in_time_filtering(
    ingested: IngestedFixture,
) -> None:
    """Projecting away available_at must not drop the filter that uses it."""
    projected = scan(ingested, as_of=at(33), instrument_ids=(BTC,), columns=("bar_start", "close"))
    assert projected.columns == ["bar_start", "close"]
    assert at(30) not in set(projected.get_column("bar_start").to_list())


def test_unknown_columns_are_rejected(ingested: IngestedFixture) -> None:
    with pytest.raises(ValueError, match="unknown bar columns"):
        scan(ingested, as_of=at(300), columns=("bar_start", "alpha"))


def test_as_of_is_required(ingested: IngestedFixture) -> None:
    """Forgetting the availability cutoff must be a construction error."""
    with pytest.raises(ValueError, match="as_of"):
        BarQuery(dataset_id=ingested.dataset_id)  # type: ignore[call-arg]


def test_the_unfiltered_scan_sees_what_the_point_in_time_scan_hides(
    ingested: IngestedFixture,
) -> None:
    """The escape hatch works, and is separately named so it cannot be reached by accident."""
    everything = ingested.catalog.scan_all_unfiltered(
        UnfilteredScan(dataset_id=ingested.dataset_id, reason="verifying the escape hatch")
    ).collect()
    assert everything.height == ingested.outcome.manifest.row_count
    assert everything.height > scan(ingested, as_of=at(33)).height


def test_the_unfiltered_scan_demands_a_stated_reason(ingested: IngestedFixture) -> None:
    with pytest.raises(ValueError):
        UnfilteredScan(dataset_id=ingested.dataset_id, reason="why")


# -- dataset identity ----------------------------------------------------------------


def test_re_ingesting_identical_source_and_policy_reuses_the_dataset_id(
    ingested: IngestedFixture,
) -> None:
    again = run_ingest(
        ingested.catalog, make_request(ingested.source, ingested.spec, ingested.policy)
    )
    assert again.manifest.dataset_id == ingested.dataset_id
    assert again.reused_existing
    assert again.manifest.content_digest == ingested.outcome.manifest.content_digest


def test_a_changed_policy_yields_a_different_dataset(ingested: IngestedFixture) -> None:
    other = ingested.policy.model_copy(update={"publication_latency": dt.timedelta(seconds=5)})
    outcome = run_ingest(ingested.catalog, make_request(ingested.source, ingested.spec, other))
    assert outcome.manifest.dataset_id != ingested.dataset_id


def test_a_corrected_source_yields_a_new_manifest_leaving_the_old_intact(
    tmp_path: Path, catalog: DatasetCatalog, spec: SyntheticSpec, policy
) -> None:
    """ARCHITECTURE.md sec. 4: provider corrections create a new dataset version."""
    original_csv = write_source_csv(tmp_path / "v1.csv", spec)
    original = run_ingest(catalog, make_request(original_csv, spec, policy))
    original_digest = original.manifest.content_digest

    corrected_csv = write_source_csv(tmp_path / "v2.csv", replace(spec, seed=spec.seed + 1))
    corrected = run_ingest(catalog, make_request(corrected_csv, spec, policy))

    assert corrected.manifest.dataset_id != original.manifest.dataset_id
    reloaded = catalog.resolve(original.manifest.dataset_id)
    assert reloaded.content_digest == original_digest
    catalog.verify(original.manifest.dataset_id)
    catalog.verify(corrected.manifest.dataset_id)


def test_stored_files_match_the_manifest(ingested: IngestedFixture) -> None:
    ingested.catalog.verify(ingested.dataset_id)


def test_tampering_with_a_partition_is_detected(ingested: IngestedFixture) -> None:
    partition = ingested.catalog.root / ingested.outcome.manifest.partitions[0].path
    partition.write_bytes(partition.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="has changed since it was validated"):
        ingested.catalog.verify(ingested.dataset_id)


def test_two_datasets_never_share_a_partition_file(
    tmp_path: Path, catalog: DatasetCatalog, spec: SyntheticSpec, policy
) -> None:
    """Regression guard: dataset versions must be isolated on disk.

    Both datasets cover the same instruments over the same UTC days. If the partition path
    did not include the dataset id they would write to the same files, and the first
    manifest would silently start describing the second dataset's bytes -- a run pinned to
    a dataset id would read data it was never validated against.
    """
    first = run_ingest(
        catalog, make_request(write_source_csv(tmp_path / "a.csv", spec), spec, policy)
    )
    second = run_ingest(
        catalog,
        make_request(
            write_source_csv(tmp_path / "b.csv", replace(spec, seed=spec.seed + 7)), spec, policy
        ),
    )
    assert first.manifest.dataset_id != second.manifest.dataset_id
    paths_a = {p.path for p in first.manifest.partitions}
    paths_b = {p.path for p in second.manifest.partitions}
    assert not paths_a & paths_b
    catalog.verify(first.manifest.dataset_id)
    catalog.verify(second.manifest.dataset_id)
