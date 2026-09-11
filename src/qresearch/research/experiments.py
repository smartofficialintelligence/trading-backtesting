"""Run index and comparison.

The index is a rebuildable projection of ``result.json`` and ``run_spec.json`` files
(ARCHITECTURE.md sec. 8): drop it and rebuild it at any time. Comparison always shows the
split role, dataset id, cost scenario, and warning count next to performance, so a
headline number is never read without its assumptions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import duckdb
import polars as pl

from qresearch.artifacts.contracts import RunResult, RunSpec
from qresearch.artifacts.local import LocalArtifactStore
from qresearch.ids import RunId
from qresearch.research.metrics import Metrics

METRIC_COLUMNS: tuple[str, ...] = (
    "periods",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "turnover_annualized",
    "avg_gross_exposure",
    "avg_net_exposure",
    "order_count",
    "fill_count",
    "period_hit_rate",
    "total_costs",
    "cost_fraction",
    "mean_participation",
    "warning_count",
)

_RUN_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.String(),
    "config_id": pl.String(),
    "status": pl.String(),
    "experiment_id": pl.String(),
    "label": pl.String(),
    "dataset_id": pl.String(),
    "bar_size": pl.String(),
    "calendar_id": pl.String(),
    "instrument_count": pl.Int64(),
    "strategy": pl.String(),
    "cost_scenario": pl.String(),
    "fill_rule": pl.String(),
    "plan": pl.String(),
    "seed": pl.Int64(),
    "code_revision": pl.String(),
    "started_at": pl.Datetime("us", "UTC"),
    "finished_at": pl.Datetime("us", "UTC"),
    "fold_count": pl.Int64(),
    "warning_count": pl.Int64(),
    "economic_digest": pl.String(),
    "annualization": pl.String(),
}


def _plan_label(spec: RunSpec) -> str:
    plan = spec.plan
    if hasattr(plan, "kind"):
        return f"{plan.kind.value} train={plan.train} test={plan.test} purge={plan.effective_purge}"
    return "fixed"


def _metric_row(metrics: Metrics | None, prefix: str) -> dict[str, Any]:
    if metrics is None:
        return {f"{prefix}{c}": None for c in METRIC_COLUMNS}
    data = metrics.model_dump(mode="python")
    return {f"{prefix}{c}": data.get(c) for c in METRIC_COLUMNS}


def run_table(
    store: LocalArtifactStore, run_ids: Sequence[RunId | str] | None = None
) -> pl.DataFrame:
    """One row per run: identity, assumptions, and per-role aggregate metrics."""
    ids = list(run_ids) if run_ids is not None else list(store.list_runs())
    rows: list[dict[str, Any]] = []
    for run_id in ids:
        spec, result = store.load_spec(run_id), store.load_result(run_id)
        row: dict[str, Any] = {
            "run_id": str(run_id),
            "config_id": spec.config_id,
            "status": result.status.value,
            "experiment_id": spec.experiment_id,
            "label": spec.label,
            "dataset_id": str(spec.dataset_id),
            "bar_size": spec.bar_size,
            "calendar_id": spec.calendar_id,
            "instrument_count": len(spec.instrument_ids),
            "strategy": spec.strategy.kind,
            "cost_scenario": spec.cost_scenario,
            "fill_rule": spec.simulation.execution.fill_rule.value,
            "plan": _plan_label(spec),
            "seed": spec.seed,
            "code_revision": spec.code_revision,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "fold_count": len({f.fold for f in result.folds}),
            "warning_count": sum(w.occurrences for w in result.warnings),
            "economic_digest": result.economic_digest,
            "annualization": next(iter(result.aggregate.values())).annualization.label
            if result.aggregate
            else None,
        }
        for role in ("validation", "test"):
            row.update(_metric_row(result.aggregate.get(role), f"{role}_"))
        rows.append(row)
    schema: dict[str, Any] = dict(_RUN_SCHEMA)
    for role in ("validation", "test"):
        for c in METRIC_COLUMNS:
            schema[f"{role}_{c}"] = (
                pl.Int64()
                if c in ("periods", "order_count", "fill_count", "warning_count")
                else pl.Float64()
            )
    return pl.DataFrame(rows, schema=schema)


def fold_table(
    store: LocalArtifactStore, run_ids: Sequence[RunId | str] | None = None
) -> pl.DataFrame:
    """One row per (run, fold, role) with that fold's metrics."""
    ids = list(run_ids) if run_ids is not None else list(store.list_runs())
    rows: list[dict[str, Any]] = []
    for run_id in ids:
        result = store.load_result(run_id)
        for fold in result.folds:
            rows.append(
                {
                    "run_id": str(run_id),
                    "fold": fold.fold,
                    "role": fold.role.value,
                    "start": fold.range.start,
                    "end": fold.range.end,
                    "decisions": fold.decisions,
                    **_metric_row(fold.metrics, ""),
                }
            )
    schema: dict[str, Any] = {
        "run_id": pl.String(),
        "fold": pl.Int64(),
        "role": pl.String(),
        "start": pl.Datetime("us", "UTC"),
        "end": pl.Datetime("us", "UTC"),
        "decisions": pl.Int64(),
    }
    for c in METRIC_COLUMNS:
        schema[c] = (
            pl.Int64()
            if c in ("periods", "order_count", "fill_count", "warning_count")
            else pl.Float64()
        )
    return pl.DataFrame(rows, schema=schema)


def build_run_index(
    store: LocalArtifactStore, connection: duckdb.DuckDBPyConnection
) -> tuple[str, ...]:
    """(Re)create ``runs`` and ``run_folds`` tables in DuckDB from the store."""
    runs = run_table(store)
    folds = fold_table(store)
    connection.register("_runs_src", runs.to_arrow())
    connection.execute("CREATE OR REPLACE TABLE runs AS SELECT * FROM _runs_src")
    connection.unregister("_runs_src")
    connection.register("_folds_src", folds.to_arrow())
    connection.execute("CREATE OR REPLACE TABLE run_folds AS SELECT * FROM _folds_src")
    connection.unregister("_folds_src")
    return ("runs", "run_folds")


COMPARE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "config_id",
    "label",
    "status",
    "dataset_id",
    "strategy",
    "cost_scenario",
    "fill_rule",
    "fold_count",
    "warning_count",
    "role",
    "sharpe",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "turnover_annualized",
    "cost_fraction",
    "period_hit_rate",
    "fill_count",
)


def compare(
    store: LocalArtifactStore, run_ids: Sequence[RunId | str], *, role: str = "test"
) -> pl.DataFrame:
    """Side-by-side view of runs for one split role, assumptions first."""
    if role not in ("validation", "test"):
        raise ValueError("role must be 'validation' or 'test'")
    table = run_table(store, run_ids)
    # The run-level warning_count (all roles, all folds) is the one to show; the per-role
    # copy would collide with it.
    renamed = table.rename(
        {
            f"{role}_{c}": c
            for c in METRIC_COLUMNS
            if c != "warning_count" and f"{role}_{c}" in table.columns
        }
    )
    return renamed.with_columns(pl.lit(role).alias("role")).select(list(COMPARE_COLUMNS))


class Agreement(StrEnum):
    EXACT = "exact"
    WITHIN_TOLERANCE = "within_tolerance"
    DIFFERENT = "different"
    MISSING = "missing"


def compare_metrics(a: Metrics, b: Metrics, *, rtol: float = 1e-9) -> dict[str, Agreement]:
    """Classify each metric: exact, tolerance-equal, or different (DEVELOPMENT_PLAN.md sec. 7)."""
    out: dict[str, Agreement] = {}
    da, db = a.model_dump(mode="python"), b.model_dump(mode="python")
    for name in METRIC_COLUMNS:
        va, vb = da.get(name), db.get(name)
        if va is None or vb is None:
            out[name] = Agreement.EXACT if va is vb else Agreement.MISSING
        elif va == vb:
            out[name] = Agreement.EXACT
        elif (
            isinstance(va, float)
            and isinstance(vb, float)
            and math.isclose(va, vb, rel_tol=rtol, abs_tol=0.0)
        ):
            out[name] = Agreement.WITHIN_TOLERANCE
        else:
            out[name] = Agreement.DIFFERENT
    return out


def results_by_experiment(store: LocalArtifactStore, experiment_id: str) -> list[RunResult]:
    return [
        store.load_result(r)
        for r in store.list_runs()
        if store.load_spec(r).experiment_id == experiment_id
    ]
