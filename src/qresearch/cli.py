"""Command-line interface.

Thin by design (ARCHITECTURE.md sec. 3): commands parse arguments, call an application
service, and format output. No business logic lives here.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from qresearch import __version__
from qresearch.application.config import load_ingest_config
from qresearch.application.ingest import DataQualityError, ingest_bars
from qresearch.data.adapters.base import IngestRequest
from qresearch.data.adapters.csv_parquet import LocalFileBarAdapter
from qresearch.data.catalog import DatasetCatalog, DatasetNotFoundError
from qresearch.data.duck import build_views, connect, query_asof
from qresearch.data.manifests import DatasetManifest, ValidationFinding, ValidationSeverity
from qresearch.data.point_in_time import BarQuery
from qresearch.ids import DatasetId
from qresearch.time import ensure_utc

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Ingest, inspect, and verify datasets.")
app.add_typer(data_app, name="data")

RootOption = Annotated[
    Path, typer.Option("--root", envvar="QRESEARCH_DATA_ROOT", help="Catalog root directory.")
]

_SEVERITY_MARK = {
    ValidationSeverity.ERROR: "ERROR  ",
    ValidationSeverity.WARNING: "WARNING",
    ValidationSeverity.INFO: "INFO   ",
}


@app.callback()
def _root() -> None:
    """qresearch -- intraday strategy research and backtesting."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@data_app.command("ingest")
def data_ingest(
    config: Annotated[Path, typer.Option("--config", "-c", help="Ingestion config YAML.")],
    root: RootOption = Path("data"),
) -> None:
    """Normalize, validate, and store a source file as an immutable dataset."""
    spec = load_ingest_config(config)
    request = IngestRequest(
        uri=spec.source_uri,
        bar_size=spec.bar_size,
        policy=spec.policy,
        instruments=spec.instruments,
        source_name=spec.source_name,
        mapping=spec.mapping,
        source_revision=spec.source_revision,
        feed=spec.feed,
    )
    try:
        outcome = ingest_bars(
            request,
            catalog=DatasetCatalog(root),
            adapter=LocalFileBarAdapter(),
            asset_class=spec.asset_class,
            venue=spec.venue,
            calendar_id=spec.calendar_id,
            normalization_version=spec.normalization_version,
            created_by=f"qresearch-cli/{__version__}",
            expect_complete_grid=spec.expect_complete_grid,
        )
    except DataQualityError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    manifest = outcome.manifest
    verb = "reused" if outcome.reused_existing else "wrote"
    typer.echo(f"{verb} {manifest.dataset_id}  rows={manifest.row_count}")
    _print_findings(outcome.report.findings)


@data_app.command("list")
def data_list(root: RootOption = Path("data")) -> None:
    """List datasets in the catalog."""
    catalog = DatasetCatalog(root)
    ids = catalog.list_datasets()
    if not ids:
        typer.echo(f"no datasets under {root}")
        return
    for dataset_id in ids:
        manifest = catalog.resolve(dataset_id)
        identity = manifest.identity
        typer.echo(
            f"{dataset_id}  {identity.asset_class.value:<7} {identity.bar_size:<4} "
            f"rows={manifest.row_count:<8} instruments={len(identity.instrument_ids):<3} "
            f"{identity.range_start:%Y-%m-%d}..{identity.range_end:%Y-%m-%d}"
        )


@data_app.command("inspect")
def data_inspect(
    dataset: Annotated[str, typer.Argument(help="Dataset id.")],
    root: RootOption = Path("data"),
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the full manifest as JSON.")
    ] = False,
) -> None:
    """Show a dataset's manifest, policy, and validation findings."""
    manifest = _resolve(root, dataset)
    if as_json:
        typer.echo(json.dumps(json.loads(manifest.model_dump_json()), indent=2, sort_keys=True))
        return

    identity = manifest.identity
    policy = identity.policy
    typer.echo(f"dataset          {manifest.dataset_id}")
    typer.echo(f"asset class      {identity.asset_class.value} @ {identity.venue}")
    typer.echo(f"bar size         {identity.bar_size}   calendar {identity.calendar_id}")
    typer.echo(f"range            {identity.range_start} .. {identity.range_end}")
    typer.echo(f"instruments      {', '.join(identity.instrument_ids)}")
    typer.echo(f"rows             {manifest.row_count} in {len(manifest.partitions)} partition(s)")
    typer.echo(f"timestamp label  {policy.timestamp_label.value}")
    typer.echo(f"publication lag  {policy.publication_latency}")
    typer.echo(f"adjustment       {policy.price_adjustment.value}")
    typer.echo(f"duplicates       {policy.duplicates.value}   revisions {policy.revisions.value}")
    typer.echo(f"content digest   {manifest.content_digest}")
    typer.echo(f"created          {manifest.created_at} by {manifest.created_by}")
    typer.echo("")
    _print_findings(manifest.validation.findings)


@data_app.command("verify")
def data_verify(
    dataset: Annotated[str, typer.Argument(help="Dataset id.")],
    root: RootOption = Path("data"),
) -> None:
    """Check that stored files still match the manifest."""
    catalog = DatasetCatalog(root)
    _resolve(root, dataset)
    try:
        catalog.verify(DatasetId(dataset))
    except (ValueError, FileNotFoundError, AssertionError) as error:
        typer.secho(f"FAILED: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    typer.secho(f"ok: {dataset} matches its manifest", fg=typer.colors.GREEN)


@data_app.command("head")
def data_head(
    dataset: Annotated[str, typer.Argument(help="Dataset id.")],
    as_of: Annotated[str, typer.Option("--as-of", help="UTC availability cutoff, ISO-8601.")],
    root: RootOption = Path("data"),
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
) -> None:
    """Show the first rows visible at a point in time.

    ``--as-of`` is required: there is no way to ask this command for data without stating
    when you are allowed to know it.
    """
    catalog = DatasetCatalog(root)
    _resolve(root, dataset)
    cutoff = _parse_as_of(as_of)
    frame = (
        catalog.scan_bars(BarQuery(dataset_id=DatasetId(dataset), as_of=cutoff))
        .head(limit)
        .collect()
    )
    typer.echo(f"as of {cutoff}  ({frame.height} row(s) shown)")
    typer.echo(str(frame))


@data_app.command("index")
def data_index(
    root: RootOption = Path("data"),
    database: Annotated[
        Path, typer.Option("--database", help="DuckDB file to build views into.")
    ] = Path("data/catalog.duckdb"),
) -> None:
    """Rebuild the DuckDB catalog views from manifests."""
    catalog = DatasetCatalog(root)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(catalog, database=database)
    names = build_views(catalog, connection)
    connection.close()
    typer.echo(f"built {len(names)} view(s)/table(s) in {database}")


@data_app.command("sql")
def data_sql(
    dataset: Annotated[str, typer.Argument(help="Dataset id.")],
    as_of: Annotated[str, typer.Option("--as-of", help="UTC availability cutoff, ISO-8601.")],
    where: Annotated[str | None, typer.Option("--where", help="Extra SQL predicate.")] = None,
    root: RootOption = Path("data"),
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Run a point-in-time query through DuckDB."""
    catalog = DatasetCatalog(root)
    _resolve(root, dataset)
    connection = connect(catalog)
    frame = query_asof(connection, DatasetId(dataset), _parse_as_of(as_of), where=where)
    typer.echo(str(frame.head(limit)))
    typer.echo(f"{frame.height} row(s) matched")


@data_app.command("demo")
def data_demo(
    root: RootOption = Path("data"),
    source_dir: Annotated[
        Path | None,
        typer.Option("--source-dir", help="Where to write the source CSV. Default: <root>/source."),
    ] = None,
) -> None:
    """Generate the synthetic two-asset sample and ingest it.

    The sample deliberately contains a gap, two late-published bars, and a revised
    duplicate, so the validation output is not empty.
    """
    source_dir = source_dir if source_dir is not None else root / "source"
    from qresearch.data.adapters.base import ColumnMapping
    from qresearch.data.contracts import AssetClass, PriceAdjustment
    from qresearch.data.manifests import (
        DuplicatePolicy,
        NormalizationPolicy,
        TimestampLabel,
    )
    from qresearch.data.synthetic import (
        SOURCE_NAME,
        SyntheticSpec,
        instruments_for,
        write_source_csv,
    )

    spec = SyntheticSpec()
    source = write_source_csv(source_dir / "synthetic_1m.csv", spec)
    typer.echo(f"wrote source {source}")

    request = IngestRequest(
        uri=str(source),
        bar_size="1m",
        policy=NormalizationPolicy(
            timestamp_label=TimestampLabel.BAR_START,
            publication_latency=_dt.timedelta(seconds=2),
            price_adjustment=PriceAdjustment.NOT_APPLICABLE,
            duplicates=DuplicatePolicy.KEEP_HIGHEST_REVISION,
            volume_unit="base_asset",
        ),
        instruments=instruments_for(spec),
        source_name=SOURCE_NAME,
        mapping=ColumnMapping(
            trade_count="trades", available_at="available_at", revision="revision"
        ),
    )
    outcome = ingest_bars(
        request,
        catalog=DatasetCatalog(root),
        adapter=LocalFileBarAdapter(),
        asset_class=AssetClass.CRYPTO,
        venue="SYNTH",
        calendar_id="24x7:1",
        normalization_version="1",
        created_by=f"qresearch-cli/{__version__}",
        expect_complete_grid=True,
    )
    typer.echo(f"dataset {outcome.manifest.dataset_id}  rows={outcome.manifest.row_count}")
    _print_findings(outcome.report.findings)
    typer.echo("")
    typer.echo(f"next: qresearch data inspect {outcome.manifest.dataset_id} --root {root}")


def _resolve(root: Path, dataset: str) -> DatasetManifest:
    try:
        return DatasetCatalog(root).resolve(DatasetId(dataset))
    except DatasetNotFoundError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error


def _parse_as_of(text: str) -> _dt.datetime:
    try:
        return ensure_utc(_dt.datetime.fromisoformat(text))
    except ValueError as error:
        typer.secho(
            f"invalid --as-of {text!r}: {error}. Provide an ISO-8601 timestamp with an "
            "offset, for example 2024-03-04T01:00:00Z",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from error


def _print_findings(findings: Sequence[ValidationFinding]) -> None:
    if not findings:
        typer.echo("no validation findings")
        return
    for finding in findings:
        colour = {
            ValidationSeverity.ERROR: typer.colors.RED,
            ValidationSeverity.WARNING: typer.colors.YELLOW,
        }.get(finding.severity)
        scope = f" [{finding.instrument_id}]" if finding.instrument_id else ""
        typer.secho(
            f"  {_SEVERITY_MARK[finding.severity]} {finding.check}{scope} "
            f"(x{finding.occurrences}): {finding.message}",
            fg=colour,
        )


if __name__ == "__main__":  # pragma: no cover
    app()
