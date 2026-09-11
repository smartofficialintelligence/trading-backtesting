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

import polars as pl
import typer

from qresearch import __version__
from qresearch.application.config import load_backtest_config, load_ingest_config
from qresearch.application.ingest import DataQualityError, ingest_bars
from qresearch.application.run_backtest import (
    COST_SCENARIOS,
    execute,
    resolve_spec,
    run_backtest,
    run_sensitivity,
)
from qresearch.artifacts.contracts import RunResult
from qresearch.artifacts.local import LocalArtifactStore, economic_digest
from qresearch.data.adapters.base import ColumnMapping, IngestRequest
from qresearch.data.adapters.csv_parquet import LocalFileBarAdapter
from qresearch.data.calendars import attach_sessions, get_calendar
from qresearch.data.catalog import DatasetCatalog, DatasetNotFoundError
from qresearch.data.contracts import PriceAdjustment
from qresearch.data.duck import build_views, connect, query_asof
from qresearch.data.manifests import (
    DatasetManifest,
    DuplicatePolicy,
    NormalizationPolicy,
    TimestampLabel,
    ValidationFinding,
    ValidationSeverity,
)
from qresearch.data.point_in_time import BarQuery, UnfilteredScan
from qresearch.data.synthetic import SOURCE_NAME, SyntheticSpec, instruments_for, write_source_csv
from qresearch.data.validation import validate_bars
from qresearch.features.pipeline import compute_features
from qresearch.features.registry import build_feature
from qresearch.ids import DatasetId, RunId
from qresearch.research.experiments import build_run_index, compare, fold_table, run_table
from qresearch.time import ensure_utc

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Ingest, inspect, and verify datasets.")
features_app = typer.Typer(no_args_is_help=True, help="Materialise feature sets.")
backtest_app = typer.Typer(no_args_is_help=True, help="Run walk-forward backtests.")
runs_app = typer.Typer(no_args_is_help=True, help="Inspect, compare, and reproduce runs.")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(backtest_app, name="backtest")
app.add_typer(runs_app, name="runs")

RootOption = Annotated[
    Path, typer.Option("--root", envvar="QRESEARCH_DATA_ROOT", help="Catalog root directory.")
]
RunsOption = Annotated[
    Path, typer.Option("--runs", envvar="QRESEARCH_RUNS_ROOT", help="Run store directory.")
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


# -- data ----------------------------------------------------------------------------------------


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


@data_app.command("validate")
def data_validate(
    dataset: Annotated[str, typer.Argument(help="Dataset id.")],
    root: RootOption = Path("data"),
    complete_grid: Annotated[
        bool, typer.Option("--complete-grid", help="Report every gap in the bar grid.")
    ] = False,
) -> None:
    """Re-run the data-quality checks against the stored dataset, with its calendar."""
    catalog = DatasetCatalog(root)
    manifest = _resolve(root, dataset)
    frame = catalog.scan_all_unfiltered(
        UnfilteredScan(dataset_id=DatasetId(dataset), reason="re-validation from the CLI")
    ).collect()
    report = validate_bars(
        frame,
        policy=manifest.identity.policy,
        expect_complete_grid=complete_grid,
        calendar=get_calendar(manifest.identity.calendar_id),
    )
    _print_findings(report.findings)
    if not report.ok:
        raise typer.Exit(code=2)


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
    market: Annotated[
        str, typer.Option("--market", help="'crypto' (24x7) or 'equity' (XNYS sessions).")
    ] = "crypto",
    source_dir: Annotated[
        Path | None,
        typer.Option("--source-dir", help="Where to write the source CSV. Default: <root>/source."),
    ] = None,
) -> None:
    """Generate a synthetic two-asset sample and ingest it.

    The sample deliberately contains a gap, late-published bars, and a revised duplicate,
    so the validation output is not empty.
    """
    if market not in ("crypto", "equity"):
        typer.secho("--market must be 'crypto' or 'equity'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    spec = SyntheticSpec() if market == "crypto" else SyntheticSpec.equity()
    source_dir = source_dir if source_dir is not None else root / "source"
    source = write_source_csv(source_dir / f"synthetic_{market}_1m.csv", spec)
    typer.echo(f"wrote source {source}")

    request = IngestRequest(
        uri=str(source),
        bar_size="1m",
        policy=NormalizationPolicy(
            timestamp_label=TimestampLabel.BAR_START,
            publication_latency=_dt.timedelta(seconds=2),
            price_adjustment=PriceAdjustment.NOT_APPLICABLE
            if market == "crypto"
            else PriceAdjustment.NONE,
            duplicates=DuplicatePolicy.KEEP_HIGHEST_REVISION,
            volume_unit="base_asset" if market == "crypto" else "shares",
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
        asset_class=spec.asset_class,
        venue=spec.venue,
        calendar_id=spec.calendar_id,
        normalization_version="1",
        created_by=f"qresearch-cli/{__version__}",
        expect_complete_grid=market == "crypto",
    )
    typer.echo(f"dataset {outcome.manifest.dataset_id}  rows={outcome.manifest.row_count}")
    _print_findings(outcome.report.findings)
    typer.echo("")
    typer.echo(f"next: qresearch data inspect {outcome.manifest.dataset_id} --root {root}")


# -- features ------------------------------------------------------------------------------------


@features_app.command("build")
def features_build(
    config: Annotated[Path, typer.Option("--config", "-c", help="Backtest config YAML.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Parquet file to write.")],
    root: RootOption = Path("data"),
) -> None:
    """Materialise the config's feature set over its dataset to Parquet.

    On-demand computation inside a run is the primary path; this exists for inspection
    and for ML workflows that want a feature table. Every row keeps its ``available_at``.
    """
    cfg = load_backtest_config(config)
    catalog = DatasetCatalog(root)
    manifest = _resolve(root, cfg.dataset_id)
    bars = catalog.scan_all_unfiltered(
        UnfilteredScan(
            dataset_id=cfg.dataset_id,
            reason="feature materialisation; rows keep their available_at",
        )
    ).collect()
    bars = attach_sessions(bars, get_calendar(manifest.identity.calendar_id))
    features = [build_feature(f.kind, f.params) for f in cfg.features]
    if not features:
        typer.secho("the config declares no features", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    frame = compute_features(bars, features)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out, compression="zstd")
    sidecar = out.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "dataset_id": cfg.dataset_id,
                "features": [f.spec.model_dump(mode="json") for f in features],
                "fingerprints": [f.spec.fingerprint for f in features],
                "rows": frame.height,
            },
            indent=2,
            sort_keys=True,
        )
    )
    typer.echo(
        f"wrote {frame.height} rows x {len(features)} feature(s) to {out} (+ {sidecar.name})"
    )


# -- backtest ------------------------------------------------------------------------------------


@backtest_app.command("run")
def backtest_run(
    config: Annotated[Path, typer.Option("--config", "-c", help="Backtest config YAML.")],
    root: RootOption = Path("data"),
    runs: RunsOption = Path("runs"),
    scenario: Annotated[
        str | None,
        typer.Option("--scenario", help="One cost scenario, or omit to run all configured."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-execute even if the run exists.")
    ] = False,
) -> None:
    """Run the configured walk-forward backtest (one run per cost scenario)."""
    cfg = load_backtest_config(config)
    catalog, store = DatasetCatalog(root), LocalArtifactStore(runs)
    try:
        if scenario is not None:
            results = {
                scenario: run_backtest(
                    cfg, catalog=catalog, store=store, cost_scenario=scenario, force=force
                )
            }
        else:
            results = run_sensitivity(cfg, catalog=catalog, store=store, force=force)
    except (DatasetNotFoundError, KeyError, ValueError) as error:
        typer.secho(f"FAILED: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    for name, result in results.items():
        typer.secho(f"== scenario {name}: {result.run_id}", bold=True)
        _print_result(result, store)
    typer.echo("")
    typer.echo("assumptions to review: docs/leakage_checklist.md")


# -- runs ----------------------------------------------------------------------------------------


@runs_app.command("list")
def runs_list(runs: RunsOption = Path("runs")) -> None:
    """List completed runs, one per line: id, label, strategy, scenario, then test metrics."""
    table = run_table(LocalArtifactStore(runs))
    if table.is_empty():
        typer.echo(f"no runs under {runs}")
        return
    for row in table.sort("started_at").iter_rows(named=True):
        typer.echo(
            f"{row['run_id']}  {row['label'] or '-':<24} {row['strategy']:<20} "
            f"{row['cost_scenario']:<9} folds={row['fold_count']} warnings={row['warning_count']}  "
            f"test: sharpe {_fmt(row['test_sharpe'], '.2f')}  return "
            f"{_fmt(row['test_total_return'], '+.3%')}  "
            f"maxdd {_fmt(row['test_max_drawdown'], '.2%')}"
        )


@runs_app.command("show")
def runs_show(
    run_id: Annotated[str, typer.Argument()],
    runs: RunsOption = Path("runs"),
) -> None:
    """Show a run's specification, assumptions, per-fold and aggregate metrics."""
    store = LocalArtifactStore(runs)
    if not store.exists(run_id):
        typer.secho(f"no run {run_id!r} under {runs}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    spec, result = store.load_spec(run_id), store.load_result(run_id)
    typer.echo(f"run              {run_id}   status {result.status.value}")
    typer.echo(f"label            {spec.label or '-'}   experiment {spec.experiment_id or '-'}")
    typer.echo(f"dataset          {spec.dataset_id}  {spec.bar_size} {spec.calendar_id}")
    typer.echo(f"instruments      {', '.join(spec.instrument_ids)}")
    typer.echo(f"strategy         {spec.strategy.kind} {spec.strategy.params}")
    typer.echo(f"features         {[f.kind for f in spec.features]}")
    typer.echo(f"transforms       {[t.kind for t in spec.transforms]}")
    typer.echo(f"cost scenario    {spec.cost_scenario}")
    typer.echo(f"code revision    {spec.code_revision or '-'}")
    typer.echo(f"economic digest  {result.economic_digest}")
    _print_result(result, store)


@runs_app.command("compare")
def runs_compare(
    run_ids: Annotated[list[str], typer.Argument()],
    runs: RunsOption = Path("runs"),
    role: Annotated[str, typer.Option("--role", help="'test' or 'validation'.")] = "test",
) -> None:
    """Compare runs side by side: assumptions first, then performance."""
    store = LocalArtifactStore(runs)
    missing = [r for r in run_ids if not store.exists(r)]
    if missing:
        typer.secho(f"unknown runs: {missing}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(_transposed(compare(store, run_ids, role=role)))


@runs_app.command("index")
def runs_index(
    runs: RunsOption = Path("runs"),
    database: Annotated[Path, typer.Option("--database")] = Path("runs/index.duckdb"),
) -> None:
    """Rebuild the DuckDB run index."""
    import duckdb

    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    names = build_run_index(LocalArtifactStore(runs), connection)
    connection.close()
    typer.echo(f"built {', '.join(names)} in {database}")


@runs_app.command("reproduce")
def runs_reproduce(
    run_id: Annotated[str, typer.Argument()],
    root: RootOption = Path("data"),
    runs: RunsOption = Path("runs"),
) -> None:
    """Re-execute a stored run from its specification and compare economics.

    Exit 0 on an exact match, 3 if the economics differ.
    """
    store = LocalArtifactStore(runs)
    if not store.exists(run_id):
        typer.secho(f"no run {run_id!r} under {runs}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    spec, stored = store.load_spec(run_id), store.load_result(run_id)
    catalog = DatasetCatalog(root)
    manifest = _resolve(root, spec.dataset_id)
    _, frames, _ = execute(spec, catalog=catalog, manifest=manifest)
    digest = economic_digest(frames)
    if digest == stored.economic_digest:
        typer.secho(
            f"exact: {run_id} reproduces economic digest {digest[:16]}...", fg=typer.colors.GREEN
        )
        return
    typer.secho(
        f"DIFFERENT: stored {str(stored.economic_digest)[:16]}... vs recomputed {digest[:16]}...",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=3)


# -- helpers -------------------------------------------------------------------------------------


def _print_result(result: RunResult, store: LocalArtifactStore) -> None:
    spec = store.load_spec(result.run_id)
    execution = spec.simulation.execution
    typer.echo(
        f"fill rule        {execution.fill_rule.value}   latency {execution.submission_latency} + "
        f"{execution.order_latency}   expiry {execution.expire_after}"
    )
    costs = execution.costs
    typer.echo(
        f"costs            half-spread {costs.half_spread_bps} bps, commission "
        f"{costs.commission_bps} bps, slippage {costs.slippage.kind.value} "
        f"coef {costs.slippage.coefficient}, participation cap {execution.participation_cap}"
    )
    for role, metrics in result.aggregate.items():
        typer.echo(
            f"[{role:<10}] return {metrics.total_return:+.4%}  ann.return "
            f"{_fmt(metrics.annualized_return, '+.2%')}  "
            f"vol {_fmt(metrics.annualized_volatility, '.2%')}  "
            f"sharpe {_fmt(metrics.sharpe, '.2f')}  maxdd {metrics.max_drawdown:.2%}  "
            f"turnover {_fmt(metrics.turnover_annualized, '.1f')}  fills {metrics.fill_count}  "
            f"costs {metrics.cost_fraction:.4%} of equity"
        )
    if result.folds:
        typer.echo(
            str(
                fold_table(store, [result.run_id]).select(
                    "fold",
                    "role",
                    "start",
                    "end",
                    "decisions",
                    "total_return",
                    "sharpe",
                    "max_drawdown",
                    "fill_count",
                )
            )
        )
    if result.aggregate:
        typer.echo(f"annualisation    {next(iter(result.aggregate.values())).annualization.label}")
    if result.warnings:
        typer.secho(
            f"warnings ({sum(w.occurrences for w in result.warnings)}):", fg=typer.colors.YELLOW
        )
        for w in result.warnings:
            scope = f" [{w.instrument_id}]" if w.instrument_id else ""
            typer.secho(f"  {w.code}{scope} x{w.occurrences}: {w.message}", fg=typer.colors.YELLOW)
    else:
        typer.echo("warnings: none")


def _table(frame: pl.DataFrame) -> str:
    """Render a frame without Polars' width truncation, so ids stay copyable."""
    with pl.Config(
        tbl_width_chars=400,
        tbl_cols=-1,
        tbl_rows=200,
        fmt_str_lengths=80,
        tbl_hide_dataframe_shape=True,
    ):
        return str(frame)


def _transposed(frame: pl.DataFrame, key: str = "run_id") -> str:
    """Metrics as rows, runs as columns: the shape that never elides an assumption."""
    ids = frame.get_column(key).to_list()
    body = frame.drop(key).select(pl.all().cast(pl.String))
    return _table(body.transpose(include_header=True, header_name="field", column_names=ids))


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


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


__all__ = ["COST_SCENARIOS", "RunId", "app", "resolve_spec"]

if __name__ == "__main__":  # pragma: no cover
    app()
