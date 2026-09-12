"""Walk-forward backtest orchestration: config -> folds -> fits -> simulations -> run.

This is the one place the pieces are composed, and it composes them in the only order
that is leak-free:

1. Bars are read for the whole span. The engine enforces availability per instant, so the
   frame may legitimately contain the future; no strategy ever sees it early.
2. Features are computed once over the whole frame. They are causal by contract and
   verified by the leakage checkers, so this equals computing them per fold.
3. For each fold, transforms are fitted on that fold's training rows only (by
   ``available_at``, minus embargoed ranges) and applied to the fold's frame.
4. The strategy is simulated over the fold's validation range (if any) and test range,
   as separate runs of the engine. Validation and test metrics are never mixed.
5. Per-role aggregates stitch the folds' period returns into one curve.

Cost scenarios are named presets; a scenario is part of the run's identity.
"""

from __future__ import annotations

import datetime as _dt
import random
from collections.abc import Mapping
from typing import Any

import polars as pl
from pydantic import Field

from qresearch.artifacts.contracts import (
    CrossSectionalRef,
    FeatureRef,
    FoldMetrics,
    RunResult,
    RunSpec,
    RunStatus,
    StrategyRef,
    TransformRef,
)
from qresearch.artifacts.environment import capture_environment
from qresearch.artifacts.local import LocalArtifactStore
from qresearch.config import FrozenModel
from qresearch.data.calendars import attach_sessions, get_calendar
from qresearch.data.catalog import DatasetCatalog
from qresearch.data.manifests import DatasetManifest
from qresearch.data.point_in_time import UnfilteredScan
from qresearch.features.cross_sectional import CrossSectionalRank, compute_cross_sectional
from qresearch.features.pipeline import compute_features
from qresearch.features.registry import build_feature, build_transform
from qresearch.features.transforms import FittedState, TrainingData, Transform
from qresearch.ids import DatasetId, InstrumentId
from qresearch.logging import get_logger, run_context
from qresearch.research.metrics import Metrics, annualization_for, compute_metrics
from qresearch.research.splits import DataSplit, SplitRole, TimeRange
from qresearch.research.trades import build_trades
from qresearch.research.walk_forward import FixedSplitPlan, Fold, WalkForwardPlan, generate_folds
from qresearch.simulation.engine import SimulationConfig, run_simulation
from qresearch.simulation.events import WarningRecord
from qresearch.simulation.execution import CostConfig, FillRule, SlippageConfig, SlippageKind
from qresearch.strategy.registry import build_strategy
from qresearch.time import as_utc_scalar, now_utc

log = get_logger(__name__)

COST_SCENARIOS: dict[str, CostConfig] = {
    "base": CostConfig(),
    "free": CostConfig.free(),
    "low": CostConfig(
        half_spread_bps=1.0,
        commission_bps=0.5,
        slippage=SlippageConfig(kind=SlippageKind.PARTICIPATION, coefficient=10.0),
    ),
    "stressed": CostConfig(
        half_spread_bps=10.0,
        commission_bps=3.0,
        slippage=SlippageConfig(kind=SlippageKind.PARTICIPATION, coefficient=100.0),
    ),
}


class BacktestConfig(FrozenModel):
    """A backtest as a researcher writes it. Resolved into a :class:`RunSpec` per scenario."""

    dataset_id: DatasetId
    instrument_ids: tuple[str, ...] | None = None
    features: tuple[FeatureRef, ...] = ()
    cross_sectional: tuple[CrossSectionalRef, ...] = ()
    """Ranks across instruments, computed after the per-instrument features."""

    transforms: tuple[TransformRef, ...] = ()
    strategy: StrategyRef
    simulation: SimulationConfig = SimulationConfig()
    plan: WalkForwardPlan | FixedSplitPlan
    span: TimeRange | None = None
    """Defaults to the dataset's full range."""

    cost_scenarios: tuple[str, ...] = Field(default=("base",), min_length=1)
    fill_rules: tuple[FillRule, ...] = Field(
        default=(FillRule.OPEN_OF_CURRENT_BAR, FillRule.NEXT_OPEN_AFTER_ELIGIBILITY),
        min_length=1,
    )
    """Every sensitivity run brackets the fill assumption. The textbook rule leads --
    it is the closer estimate of a real fill and the headline number -- and the
    conservative rule follows as the other side of the bracket. They differ by exactly
    one bar's move on each signal fill; if that gap is most of the return, the edge lives
    inside the bar after the signal and bar data cannot settle it."""

    seed: int = 0
    evaluate_validation: bool = True
    experiment_id: str | None = None
    label: str | None = None


def resolve_spec(
    config: BacktestConfig,
    manifest: DatasetManifest,
    *,
    cost_scenario: str,
    code_revision: str | None,
    fill_rule: FillRule | None = None,
) -> RunSpec:
    if cost_scenario not in COST_SCENARIOS:
        raise KeyError(f"unknown cost scenario {cost_scenario!r}; known: {sorted(COST_SCENARIOS)}")
    universe = tuple(sorted(config.instrument_ids or manifest.identity.instrument_ids))
    unknown = sorted(set(universe) - set(manifest.identity.instrument_ids))
    if unknown:
        raise ValueError(f"instruments {unknown} are not in dataset {manifest.dataset_id}")
    fingerprints = tuple(build_feature(f.kind, f.params).spec.fingerprint for f in config.features)
    execution = config.simulation.execution.model_copy(
        update={
            "costs": COST_SCENARIOS[cost_scenario],
            "fill_rule": fill_rule or config.simulation.execution.fill_rule,
        }
    )
    simulation = config.simulation.model_copy(update={"execution": execution})
    return RunSpec(
        dataset_id=manifest.dataset_id,
        instrument_ids=universe,
        bar_size=manifest.identity.bar_size,
        calendar_id=manifest.identity.calendar_id,
        features=config.features,
        cross_sectional=config.cross_sectional,
        feature_fingerprints=fingerprints,
        transforms=config.transforms,
        strategy=config.strategy,
        simulation=simulation,
        plan=config.plan,
        span=config.span
        or TimeRange(start=manifest.identity.range_start, end=manifest.identity.range_end),
        cost_scenario=cost_scenario,
        seed=config.seed,
        evaluate_validation=config.evaluate_validation,
        code_revision=code_revision,
        experiment_id=config.experiment_id,
        label=config.label,
    )


def run_backtest(
    config: BacktestConfig,
    *,
    catalog: DatasetCatalog,
    store: LocalArtifactStore,
    cost_scenario: str = "base",
    fill_rule: FillRule | None = None,
    force: bool = False,
) -> RunResult:
    """Execute (or reuse) one run. Failures are recorded and re-raised."""
    manifest = catalog.resolve(config.dataset_id)
    environment = capture_environment(seed=config.seed)
    spec = resolve_spec(
        config,
        manifest,
        cost_scenario=cost_scenario,
        code_revision=environment.code_revision,
        fill_rule=fill_rule,
    )
    if store.exists(spec.run_id) and not force:
        existing = store.load_result(spec.run_id)
        if existing.status is RunStatus.COMPLETE:
            return existing
    handle = store.begin(spec, environment)
    try:
        result, frames, fitted = execute(
            spec, catalog=catalog, manifest=manifest, started_at=handle.started_at
        )
    except Exception as error:
        store.fail(handle, error)
        raise
    return store.finalize(handle, result, frames, fitted)


def run_sensitivity(
    config: BacktestConfig,
    *,
    catalog: DatasetCatalog,
    store: LocalArtifactStore,
    force: bool = False,
) -> dict[tuple[str, FillRule], RunResult]:
    """One run per (cost scenario, fill rule): the assumption bracket every study carries."""
    return {
        (scenario, rule): run_backtest(
            config,
            catalog=catalog,
            store=store,
            cost_scenario=scenario,
            fill_rule=rule,
            force=force,
        )
        for scenario in config.cost_scenarios
        for rule in config.fill_rules
    }


# -- execution -------------------------------------------------------------------------------------


def execute(
    spec: RunSpec,
    *,
    catalog: DatasetCatalog,
    manifest: DatasetManifest,
    started_at: _dt.datetime | None = None,
) -> tuple[RunResult, dict[str, pl.DataFrame], list[FittedState]]:
    """Run every fold and role without touching the store."""
    random.seed(spec.seed)
    started_at = started_at or now_utc()
    with run_context(spec.run_id):
        return _execute(spec, catalog=catalog, manifest=manifest, started_at=started_at)


def _execute(
    spec: RunSpec,
    *,
    catalog: DatasetCatalog,
    manifest: DatasetManifest,
    started_at: _dt.datetime,
) -> tuple[RunResult, dict[str, pl.DataFrame], list[FittedState]]:
    log.info(
        "run started",
        extra={
            "fields": {
                "dataset": spec.dataset_id,
                "strategy": spec.strategy.kind,
                "scenario": spec.cost_scenario,
                "instruments": len(spec.instrument_ids),
            }
        },
    )
    calendar = get_calendar(spec.calendar_id)
    instruments = {i: manifest.instrument(i) for i in spec.instrument_ids}
    annualization = annualization_for(spec.calendar_id, spec.bar_size)

    _check_strategy_features(spec, features_columns := _feature_columns(spec))
    del features_columns
    bars = _load_bars(catalog, spec)
    bars = attach_sessions(bars, calendar)
    features_all = _compute_features(bars, spec) if spec.features else None
    transforms = [build_transform(t.kind, t.columns, t.params) for t in spec.transforms]
    folds = (
        spec.plan.fold()
        if isinstance(spec.plan, FixedSplitPlan)
        else generate_folds(spec.plan, spec.span)
    )
    folds = folds if isinstance(folds, list) else [folds]

    fold_metrics: list[FoldMetrics] = []
    ledgers: dict[str, list[pl.DataFrame]] = {}
    fitted: list[FittedState] = []
    warnings: dict[tuple[str, str | None], WarningRecord] = {}
    role_curves: dict[SplitRole, list[pl.DataFrame]] = {}
    role_fills: dict[SplitRole, list[pl.DataFrame]] = {}

    for fold in folds:
        fold_features = features_all
        if features_all is not None and transforms:
            fold_features, states = _fit_and_apply(features_all, transforms, fold)
            fitted.extend(states)
        fold_bars = bars.filter(fold.span.predicate("bar_start"))
        fold_feats = (
            fold_features.filter(fold.span.predicate("bar_start"))
            if fold_features is not None
            else None
        )

        roles: list[tuple[SplitRole, TimeRange]] = []
        if fold.validation is not None and spec.evaluate_validation:
            roles.append((SplitRole.VALIDATION, fold.validation))
        roles.append((SplitRole.TEST, fold.test))
        for role, decision_range in roles:
            strategy = build_strategy(spec.strategy.kind, spec.strategy.params)
            result = run_simulation(
                fold_bars,
                strategy=strategy,
                instruments=instruments,
                config=spec.simulation,
                decision_range=decision_range,
                features=fold_feats,
                run_id=spec.run_id,
                fold=fold.index,
            )
            frames = result.frames()
            marks = {
                str(r["instrument_id"]): float(r["mark"])
                for r in frames["positions"].sort("at").iter_rows(named=True)
            }
            frames["trades"] = build_trades(frames["fills"], marks=marks)
            frames = _tag(frames, fold.index, role)
            for name, frame in frames.items():
                ledgers.setdefault(name, []).append(frame)
            metrics = compute_metrics(frames, bar_size=spec.bar_size, annualization=annualization)
            fold_metrics.append(
                FoldMetrics(
                    fold=fold.index,
                    role=role,
                    range=decision_range,
                    decisions=result.decisions,
                    metrics=metrics,
                )
            )
            role_curves.setdefault(role, []).append(frames["equity_curve"])
            role_fills.setdefault(role, []).append(frames["fills"])
            for w in result.warnings:
                key = (w.code, w.instrument_id)
                prior = warnings.get(key)
                warnings[key] = (
                    w
                    if prior is None
                    else prior.model_copy(update={"occurrences": prior.occurrences + w.occurrences})
                )
            log.info(
                "fold complete",
                extra={
                    "fields": {
                        "fold": fold.index,
                        "role": role.value,
                        "decisions": result.decisions,
                        "fills": len(result.fills),
                        "return": metrics.total_return,
                    }
                },
            )

    for limitation in _limitations(spec):
        warnings.setdefault((limitation.code, None), limitation)

    aggregate = {
        role.value: compute_metrics(
            {
                "equity_curve": _stitch(curves),
                "fills": pl.concat(role_fills[role]),
                "warnings": pl.DataFrame(
                    {"occurrences": [w.occurrences for w in warnings.values()]}
                ),
                "orders": pl.concat(ledgers["orders"]),
            },
            bar_size=spec.bar_size,
            annualization=annualization,
        )
        for role, curves in role_curves.items()
    }
    frames_out = {name: pl.concat(parts) for name, parts in ledgers.items()}
    frames_out["folds"] = _fold_table(folds)
    run_result = RunResult(
        run_id=spec.run_id,
        status=RunStatus.RUNNING,
        started_at=started_at,
        folds=tuple(fold_metrics),
        aggregate=aggregate,
        warnings=tuple(sorted(warnings.values(), key=lambda w: (w.code, w.instrument_id or ""))),
    )
    log.info(
        "run complete",
        extra={
            "fields": {
                "folds": len(folds),
                "warnings": sum(w.occurrences for w in warnings.values()),
            }
        },
    )
    return run_result, frames_out, fitted


def _limitations(spec: RunSpec) -> list[WarningRecord]:
    """Known modelling limitations every run must carry (DEVELOPMENT_PLAN.md Stage 5)."""
    at = spec.span.start
    costs = spec.simulation.execution.costs
    records = [
        WarningRecord(
            at=at,
            code="static_universe",
            message=(
                f"the universe is a static list of {len(spec.instrument_ids)} instrument(s) "
                "applied across the whole span; point-in-time membership is not modelled, "
                "so results can carry survivorship bias"
            ),
        )
    ]
    if costs.half_spread_bps > 0:
        records.append(
            WarningRecord(
                at=at,
                code="assumed_spread",
                message=(
                    f"a {costs.half_spread_bps:g} bps half-spread is an assumption; no quote "
                    "data was used"
                ),
            )
        )
    else:
        records.append(
            WarningRecord(
                at=at,
                code="zero_spread_assumed",
                message="fills pay no spread; this is a sensitivity baseline, not a forecast",
            )
        )
    return records


def _feature_columns(spec: RunSpec) -> set[str]:
    """Names the configured features and ranks will produce, without computing them."""
    names = {build_feature(f.kind, f.params).spec.name for f in spec.features}
    return names | {f"{r.column}_xrank" for r in spec.cross_sectional}


def _check_strategy_features(spec: RunSpec, available: set[str]) -> None:
    """Refuse a strategy that reads a feature the run does not produce.

    A rule whose comparisons reference a missing column evaluates to false everywhere, so
    the run succeeds with zero trades and no error. That is indistinguishable from a
    strategy that simply found nothing, and it is the failure this check exists to
    prevent.
    """
    strategy = build_strategy(spec.strategy.kind, spec.strategy.params)
    required = getattr(strategy, "required_features", None)
    if not required:
        return
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(
            f"the strategy reads {missing}, which this run does not produce. "
            f"Available: {sorted(available)}. A rule referencing a missing feature is "
            "silently false everywhere, so this is refused rather than run."
        )


def _load_bars(catalog: DatasetCatalog, spec: RunSpec) -> pl.DataFrame:
    # Unfiltered on purpose: the simulator publishes each row at its own available_at.
    return (
        catalog.scan_all_unfiltered(
            UnfilteredScan(
                dataset_id=spec.dataset_id,
                reason="simulation timeline; availability is enforced per instant by the engine",
                instrument_ids=tuple(InstrumentId(i) for i in spec.instrument_ids),
            )
        )
        .filter(spec.span.predicate("bar_start"))
        .collect()
    )


def _compute_features(bars: pl.DataFrame, spec: RunSpec) -> pl.DataFrame:
    features = [build_feature(f.kind, f.params) for f in spec.features]
    frame = compute_features(bars, features)
    if not spec.cross_sectional:
        return frame
    ranks = [
        CrossSectionalRank(
            column=r.column, min_members=r.min_members, pct=r.pct, descending=r.descending
        )
        for r in spec.cross_sectional
    ]
    missing = sorted({r.column for r in ranks} - set(frame.columns))
    if missing:
        raise ValueError(
            f"cross-sectional ranks reference columns {missing} that no feature produces; "
            f"available: "
            f"{sorted(set(frame.columns) - {'instrument_id', 'bar_start', 'available_at'})}"
        )
    return compute_cross_sectional(frame, ranks)


def _fit_and_apply(
    features: pl.DataFrame, transforms: list[Transform], fold: Fold
) -> tuple[pl.DataFrame, list[FittedState]]:
    train_split = DataSplit(role=SplitRole.TRAIN, range=fold.train, fold=fold.index)
    train_rows = features.filter(fold.training_predicate())
    frame = features
    states = []
    for transform in transforms:
        state = transform.fit(TrainingData(frame=train_rows, split=train_split))
        states.append(state)
        frame = transform.apply(frame, state)
        train_rows = transform.apply(train_rows, state)  # later transforms see earlier ones
    return frame, states


def _tag(frames: Mapping[str, pl.DataFrame], fold: int, role: SplitRole) -> dict[str, pl.DataFrame]:
    return {
        name: frame.with_columns(
            pl.lit(fold, dtype=pl.Int32).alias("fold"), pl.lit(role.value).alias("role")
        )
        for name, frame in frames.items()
    }


def _stitch(curves: list[pl.DataFrame]) -> pl.DataFrame:
    """Chain fold curves so each fold starts where the previous ended."""
    parts = []
    scale = 1.0
    for curve in sorted(curves, key=lambda c: as_utc_scalar(c.get_column("at").min())):
        if curve.is_empty():
            continue
        start = float(curve.item(0, "equity"))
        factor = scale / start if start else 1.0
        parts.append(
            curve.with_columns(
                (pl.col("equity") * factor).alias("equity"),
                (pl.col("gross_exposure") * factor).alias("gross_exposure"),
                (pl.col("net_exposure") * factor).alias("net_exposure"),
                (pl.col("cash") * factor).alias("cash"),
            )
        )
        scale = float(parts[-1].item(parts[-1].height - 1, "equity"))
    if not parts:
        return curves[0]
    first_start = float(curves[0].item(0, "equity")) if curves and not curves[0].is_empty() else 1.0
    stitched = pl.concat(parts)
    return stitched.with_columns(
        *[
            (pl.col(c) * first_start).alias(c)
            for c in ("equity", "gross_exposure", "net_exposure", "cash")
        ]
    )


def _fold_table(folds: list[Fold]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for split in fold.splits():
            rows.append(
                {
                    "fold": fold.index,
                    "role": split.role.value,
                    "start": split.range.start,
                    "end": split.range.end,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "fold": pl.Int32,
            "role": pl.String,
            "start": pl.Datetime("us", "UTC"),
            "end": pl.Datetime("us", "UTC"),
        },
    )


__all__ = [
    "COST_SCENARIOS",
    "BacktestConfig",
    "Metrics",
    "execute",
    "resolve_spec",
    "run_backtest",
    "run_sensitivity",
]
