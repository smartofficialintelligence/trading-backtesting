"""DuckDB views are a rebuildable projection of the Parquet truth."""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest
from tests.conftest import IngestedFixture

from qresearch.data.duck import _view_suffix, build_views, connect, query_asof

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def conn(ingested: IngestedFixture) -> duckdb.DuckDBPyConnection:
    return connect(ingested.catalog)


def test_the_catalog_table_describes_the_dataset(
    conn: duckdb.DuckDBPyConnection, ingested: IngestedFixture
) -> None:
    row = conn.execute(
        "SELECT dataset_id, bar_size, row_count, timestamp_label, error_count FROM datasets"
    ).fetchone()
    assert row is not None
    assert row[0] == ingested.dataset_id
    assert row[1] == "1m"
    assert row[2] == ingested.outcome.manifest.row_count
    assert row[3] == "bar_start"
    assert row[4] == 0


def test_the_plain_view_returns_every_row(
    conn: duckdb.DuckDBPyConnection, ingested: IngestedFixture
) -> None:
    suffix = _view_suffix(ingested.dataset_id)
    count = conn.execute(f"SELECT count(*) FROM bars_{suffix}").fetchone()
    assert count is not None
    assert count[0] == ingested.outcome.manifest.row_count


def test_the_asof_macro_applies_the_availability_filter(
    conn: duckdb.DuckDBPyConnection, ingested: IngestedFixture
) -> None:
    """SQL users get the point-in-time predicate without hand-writing it."""
    as_of = T0 + dt.timedelta(minutes=33)
    frame = query_asof(conn, ingested.dataset_id, as_of, where="instrument_id = 'CRYPTO:BTCUSD'")
    starts = set(frame.get_column("bar_start").to_list())
    assert T0 + dt.timedelta(minutes=30) not in starts
    assert T0 + dt.timedelta(minutes=31) in starts


def test_the_asof_macro_agrees_with_the_polars_scan(
    conn: duckdb.DuckDBPyConnection, ingested: IngestedFixture
) -> None:
    """The two read paths must not disagree about what was visible."""
    from qresearch.data.point_in_time import BarQuery

    as_of = T0 + dt.timedelta(minutes=90)
    sql_count = query_asof(conn, ingested.dataset_id, as_of).height
    polars_count = (
        ingested.catalog.scan_bars(BarQuery(dataset_id=ingested.dataset_id, as_of=as_of))
        .collect()
        .height
    )
    assert sql_count == polars_count


def test_partitions_are_listed(conn: duckdb.DuckDBPyConnection, ingested: IngestedFixture) -> None:
    count = conn.execute("SELECT count(*) FROM dataset_partitions").fetchone()
    assert count is not None
    assert count[0] == len(ingested.outcome.manifest.partitions)


def test_views_are_rebuildable_and_idempotent(ingested: IngestedFixture) -> None:
    connection = duckdb.connect(":memory:")
    first = build_views(ingested.catalog, connection)
    second = build_views(ingested.catalog, connection)
    assert first == second
    count = connection.execute("SELECT count(*) FROM datasets").fetchone()
    assert count is not None
    assert count[0] == 1, "rebuilding must replace the catalog table, not append to it"


def test_sql_and_polars_schemas_agree(
    conn: duckdb.DuckDBPyConnection, ingested: IngestedFixture
) -> None:
    """Hive path segments must not leak in as extra columns."""
    from qresearch.data.catalog import BAR_SCHEMA

    frame = query_asof(conn, ingested.dataset_id, T0 + dt.timedelta(minutes=5))
    assert frame.columns == BAR_SCHEMA.names
