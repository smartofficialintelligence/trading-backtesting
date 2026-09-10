"""DuckDB views over the Parquet catalog.

DuckDB is a query layer, never the source of truth (ARCHITECTURE.md sec. 4). Everything
here is derived from manifests and Parquet files and can be dropped and rebuilt at any
time, so the database file is disposable and is gitignored.

Two views are created per dataset plus two catalog-wide ones:

``bars_<dataset_id>``
    Every row in the dataset. **Not** point-in-time. Named plainly rather than
    reassuringly, because a SQL user has no availability filter applied for them.

``bars_asof_<dataset_id>``
    A table macro taking an ``as_of`` argument and applying the availability filter, so
    ad-hoc SQL has a correct default available without hand-writing the predicate.

Prefer ``.pl()`` or ``.arrow()`` over ``.fetchall()`` for anything larger than a glance:
the Arrow path hands columns straight to Polars without building a Python object per
cell. ``.fetchall()`` does work -- ``pytz`` is a dependency because DuckDB uses it to
materialise ``TIMESTAMPTZ`` into Python ``datetime`` objects and to bind them as
parameters -- it is just the slow path.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable
from pathlib import Path

import duckdb
import polars as pl

from qresearch.data.catalog import DatasetCatalog
from qresearch.data.manifests import DatasetManifest
from qresearch.ids import DatasetId
from qresearch.time import ensure_utc


def sql_utc(value: _dt.datetime) -> str:
    """Format an aware UTC datetime for interpolation into DuckDB SQL text.

    Needed where a parameter cannot be bound -- ``connection.sql()`` takes no parameters,
    unlike ``execute()`` -- and the value has to appear in the statement itself.
    """
    return ensure_utc(value).isoformat()


def _view_suffix(dataset_id: DatasetId) -> str:
    """A SQL-safe suffix. Dataset ids are ``ds_<hex>``, so only the prefix needs care."""
    return dataset_id.replace("-", "_").replace(".", "_")


def build_views(
    catalog: DatasetCatalog,
    connection: duckdb.DuckDBPyConnection,
    *,
    dataset_ids: Iterable[DatasetId] | None = None,
) -> tuple[str, ...]:
    """Create or replace views for the given datasets. Returns the view names created."""
    ids = tuple(dataset_ids) if dataset_ids is not None else catalog.list_datasets()
    created: list[str] = []
    manifests: list[DatasetManifest] = []

    for dataset_id in ids:
        manifest = catalog.resolve(dataset_id)
        manifests.append(manifest)
        paths = [str((catalog.root / p.path).resolve()) for p in manifest.partitions]
        if not paths:
            continue
        suffix = _view_suffix(dataset_id)
        files = ", ".join(f"'{p}'" for p in paths)

        # hive_partitioning=false: the path segments (asset_class=, dataset=, date=...)
        # are layout, not data. Letting DuckDB surface them as columns would make the SQL
        # schema drift from the Polars one and shadow the real bar_size column.
        connection.execute(
            f"CREATE OR REPLACE VIEW bars_{suffix} AS "
            f"SELECT * FROM read_parquet([{files}], hive_partitioning=false)"
        )
        created.append(f"bars_{suffix}")

        # A macro rather than a view so the cutoff is a parameter the caller must supply.
        connection.execute(
            f"CREATE OR REPLACE MACRO bars_asof_{suffix}(as_of) AS TABLE "
            f"SELECT * FROM bars_{suffix} WHERE available_at <= as_of"
        )
        created.append(f"bars_asof_{suffix}")

    _build_catalog_tables(connection, catalog.root, manifests)
    created.extend(("datasets", "dataset_partitions"))
    return tuple(created)


def _build_catalog_tables(
    connection: duckdb.DuckDBPyConnection, root: Path, manifests: list[DatasetManifest]
) -> None:
    """Materialise manifest metadata so datasets are discoverable from SQL."""
    connection.execute("""
        CREATE OR REPLACE TABLE datasets (
            dataset_id TEXT PRIMARY KEY,
            asset_class TEXT, venue TEXT, bar_size TEXT, calendar_id TEXT,
            instrument_count INTEGER, row_count BIGINT,
            range_start TIMESTAMPTZ, range_end TIMESTAMPTZ,
            timestamp_label TEXT, publication_latency INTERVAL,
            price_adjustment TEXT, duplicates TEXT, revisions TEXT,
            normalization_version TEXT, content_digest TEXT,
            created_at TIMESTAMPTZ, created_by TEXT,
            error_count INTEGER, warning_count INTEGER
        )
    """)
    connection.execute("""
        CREATE OR REPLACE TABLE dataset_partitions (
            dataset_id TEXT, path TEXT, row_count BIGINT, byte_size BIGINT,
            min_bar_start TIMESTAMPTZ, max_bar_start TIMESTAMPTZ,
            min_available_at TIMESTAMPTZ, max_available_at TIMESTAMPTZ
        )
    """)
    for manifest in manifests:
        identity = manifest.identity
        connection.execute(
            "INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                manifest.dataset_id,
                identity.asset_class.value,
                identity.venue,
                identity.bar_size,
                identity.calendar_id,
                len(identity.instrument_ids),
                manifest.row_count,
                identity.range_start,
                identity.range_end,
                identity.policy.timestamp_label.value,
                identity.policy.publication_latency,
                identity.policy.price_adjustment.value,
                identity.policy.duplicates.value,
                identity.policy.revisions.value,
                identity.normalization_version,
                manifest.content_digest,
                manifest.created_at,
                manifest.created_by,
                len(manifest.validation.errors),
                len(manifest.validation.warnings),
            ],
        )
        for partition in manifest.partitions:
            connection.execute(
                "INSERT INTO dataset_partitions VALUES (?,?,?,?,?,?,?,?)",
                [
                    manifest.dataset_id,
                    partition.path,
                    partition.row_count,
                    partition.byte_size,
                    partition.min_bar_start,
                    partition.max_bar_start,
                    partition.min_available_at,
                    partition.max_available_at,
                ],
            )


def connect(
    catalog: DatasetCatalog, *, database: str | Path = ":memory:"
) -> duckdb.DuckDBPyConnection:
    """Open a connection with views for every dataset in the catalog already built."""
    connection = duckdb.connect(str(database))
    build_views(catalog, connection)
    return connection


def query_asof(
    connection: duckdb.DuckDBPyConnection,
    dataset_id: DatasetId,
    as_of: _dt.datetime,
    *,
    where: str | None = None,
) -> pl.DataFrame:
    """Point-in-time result for one dataset, returned as a Polars frame.

    Args:
        where: an optional additional SQL predicate. It is interpolated, so it must not
            carry untrusted input -- this is a local research tool, not a served API.
    """
    suffix = _view_suffix(dataset_id)
    clause = f" WHERE {where}" if where else ""
    return connection.sql(
        f"SELECT * FROM bars_asof_{suffix}('{sql_utc(as_of)}'::TIMESTAMPTZ){clause}"
    ).pl()
