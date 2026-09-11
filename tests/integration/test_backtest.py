"""End to end: ingest the synthetic sample, run a walk-forward backtest, persist, compare."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pytest
import yaml
from tests.conftest import IngestedFixture

from qresearch.application.config import load_backtest_config
from qresearch.application.run_backtest import (
    COST_SCENARIOS,
    BacktestConfig,
    resolve_spec,
    run_backtest,
    run_sensitivity,
)
from qresearch.artifacts.contracts import FeatureRef, RunStatus, StrategyRef, TransformRef
from qresearch.artifacts.local import LocalArtifactStore
from qresearch.research.experiments import (
    Agreement,
    build_run_index,
    compare,
    compare_metrics,
    fold_table,
    run_table,
)
from qresearch.research.walk_forward import WalkForwardPlan
from qresearch.simulation.engine import SimulationConfig
from qresearch.simulation.execution import ExecutionConfig, FillRule

T0 = dt.datetime(2024, 3, 4, tzinfo=dt.UTC)
H = dt.timedelta(hours=1)
M = dt.timedelta(minutes=1)


def config(**overrides: object) -> BacktestConfig:
    base: dict[str, object] = {
        "dataset_id": "placeholder",
        "features": (
            FeatureRef(kind="lagged_return", params={"lag": 1}),
            FeatureRef(kind="rolling_volatility", params={"window": 5}),
        ),
        "transforms": (TransformRef(kind="zscore", columns=("ret_1",)),),
        "strategy": StrategyRef(kind="lagged_signal", params={"feature": "ret_1", "weight": 0.4}),
        "simulation": SimulationConfig(
            initial_cash=10_000.0,
            execution=ExecutionConfig(
                liquidity_lookback_bars=5, expire_after=dt.timedelta(minutes=3)
            ),
        ),
        "plan": WalkForwardPlan(
            train=H, validation=30 * M, test=30 * M, purge=2 * M, warmup=10 * M
        ),
        "cost_scenarios": ("base", "free", "stressed"),
        "fill_rules": (FillRule.NEXT_OPEN_AFTER_ELIGIBILITY,),
        "experiment_id": "exp-1",
        "label": "smoke",
    }
    return BacktestConfig.model_validate(base | overrides)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "runs")


def test_walk_forward_run_end_to_end(ingested: IngestedFixture, store: LocalArtifactStore) -> None:
    cfg = config(dataset_id=ingested.dataset_id)
    result = run_backtest(cfg, catalog=ingested.catalog, store=store)

    assert result.status is RunStatus.COMPLETE
    assert store.exists(result.run_id)
    roles = {(f.fold, f.role.value) for f in result.folds}
    assert len({f.fold for f in result.folds}) == 2, (
        "4h of data: two 30m test folds after 1h train + gaps"
    )
    assert all((k, "validation") in roles and (k, "test") in roles for k in {f for f, _ in roles})
    assert set(result.aggregate) == {"validation", "test"}
    assert result.economic_digest

    # Every fold's transform was fitted on that fold's training range only.
    states = store.load_fitted_states(result.run_id)
    assert [s.fold for s in states] == sorted({f.fold for f in result.folds})
    for state in states:
        assert state.transform == "zscore" and state.fitted_on.duration == H

    # Ledgers carry fold and role, and the run's feature frame never leaked ahead.
    fills = store.load_frame(result.run_id, "fills")
    assert {"fold", "role"} <= set(fills.columns)
    orders = store.load_frame(result.run_id, "orders")
    assert (orders["signal_at"] <= orders["order_at"]).all() and (
        orders["order_at"] <= orders["eligible_at"]
    ).all()
    folds = store.load_frame(result.run_id, "folds")
    assert set(folds["role"].to_list()) >= {"warmup", "train", "purge", "validation", "test"}


def test_identical_rerun_is_reused_not_recomputed(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    cfg = config(dataset_id=ingested.dataset_id)
    first = run_backtest(cfg, catalog=ingested.catalog, store=store)
    second = run_backtest(cfg, catalog=ingested.catalog, store=store)
    assert second == first
    forced = run_backtest(cfg, catalog=ingested.catalog, store=store, force=True)
    assert forced.economic_digest == first.economic_digest, (
        "deterministic: same economics on a forced rerun"
    )
    assert store.list_runs() == (first.run_id,)


def test_cost_scenarios_are_separate_runs_ordered_as_expected(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    results = run_sensitivity(
        config(dataset_id=ingested.dataset_id), catalog=ingested.catalog, store=store
    )
    results = {k[0]: v for k, v in results.items()}
    assert set(results) == {"base", "free", "stressed"}
    assert len({r.run_id for r in results.values()}) == 3
    test = {k: v.aggregate["test"] for k, v in results.items()}
    assert test["free"].total_costs == 0.0
    assert test["free"].total_return >= test["base"].total_return >= test["stressed"].total_return
    assert test["stressed"].cost_fraction > test["base"].cost_fraction > 0


def test_validation_and_test_metrics_are_separate_and_labelled(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    result = run_backtest(
        config(dataset_id=ingested.dataset_id), catalog=ingested.catalog, store=store
    )
    table = fold_table(store, [result.run_id])
    assert set(table["role"].to_list()) == {"validation", "test"}
    for fold in result.folds:
        assert fold.metrics.annualization.label.startswith("24x7")
    view = compare(store, [result.run_id], role="test")
    assert view.columns[:10] == [
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
    ], "assumptions lead; performance follows"
    assert view.item(0, "role") == "test" and view.item(0, "dataset_id") == ingested.dataset_id


def test_run_index_is_rebuildable_in_duckdb(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    run_sensitivity(
        config(dataset_id=ingested.dataset_id, cost_scenarios=("base", "free")),
        catalog=ingested.catalog,
        store=store,
    )
    conn = duckdb.connect()
    build_run_index(store, conn)
    build_run_index(store, conn)
    rows = conn.sql(
        "SELECT cost_scenario, test_sharpe, warning_count FROM runs ORDER BY cost_scenario"
    ).pl()
    assert rows.get_column("cost_scenario").to_list() == ["base", "free"]
    assert conn.sql("SELECT count(*) FROM run_folds").pl().item() == 2 * 4
    assert run_table(store).height == 2


def test_a_failing_run_is_recorded_and_raised(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    cfg = config(
        dataset_id=ingested.dataset_id,
        strategy=StrategyRef(kind="lagged_signal", params={"feature": "nope"}),
    )
    # 'nope' is not a feature column: the strategy sees None and never trades -- not a failure.
    ok = run_backtest(cfg, catalog=ingested.catalog, store=store)
    assert ok.aggregate["test"].fill_count == 0

    bad = config(
        dataset_id=ingested.dataset_id,
        transforms=(TransformRef(kind="zscore", columns=("missing",)),),
    )
    with pytest.raises(ValueError, match="columns \\['missing'\\] are not present"):
        run_backtest(bad, catalog=ingested.catalog, store=store)
    assert not store.exists(
        resolve_spec(
            bad, ingested.catalog.resolve(bad.dataset_id), cost_scenario="base", code_revision=None
        ).run_id
    )
    assert list((store.root / "failed").iterdir())


def test_unknown_instrument_or_scenario_is_refused(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    with pytest.raises(ValueError, match="not in dataset"):
        run_backtest(
            config(dataset_id=ingested.dataset_id, instrument_ids=("CRYPTO:NOPE",)),
            catalog=ingested.catalog,
            store=store,
        )
    with pytest.raises(KeyError, match="unknown cost scenario"):
        run_backtest(
            config(dataset_id=ingested.dataset_id),
            catalog=ingested.catalog,
            store=store,
            cost_scenario="wild",
        )


def test_spec_identity_reflects_every_economic_input(ingested: IngestedFixture) -> None:
    manifest = ingested.catalog.resolve(ingested.dataset_id)
    base = resolve_spec(
        config(dataset_id=ingested.dataset_id), manifest, cost_scenario="base", code_revision="abc"
    )
    assert (
        base.run_id
        == resolve_spec(
            config(dataset_id=ingested.dataset_id, label="x"),
            manifest,
            cost_scenario="base",
            code_revision="abc",
        ).run_id
    )
    assert (
        base.run_id
        != resolve_spec(
            config(dataset_id=ingested.dataset_id),
            manifest,
            cost_scenario="free",
            code_revision="abc",
        ).run_id
    )
    assert (
        base.run_id
        != resolve_spec(
            config(dataset_id=ingested.dataset_id),
            manifest,
            cost_scenario="base",
            code_revision="def",
        ).run_id
    )
    changed = config(
        dataset_id=ingested.dataset_id,
        features=(FeatureRef(kind="lagged_return", params={"lag": 2}),),
    )
    assert (
        base.run_id
        != resolve_spec(changed, manifest, cost_scenario="base", code_revision="abc").run_id
    )
    assert base.simulation.execution.costs == COST_SCENARIOS["base"]


def test_backtest_config_loads_from_yaml(tmp_path: Path, ingested: IngestedFixture) -> None:
    text = yaml.safe_dump(
        {
            "dataset_id": ingested.dataset_id,
            "features": [{"kind": "lagged_return", "params": {"lag": 1}}],
            "strategy": {"kind": "buy_and_hold", "params": {"weights": {"CRYPTO:BTCUSD": 0.5}}},
            "simulation": {"initial_cash": 5000},
            "plan": {"kind": "rolling", "train": "PT1H", "test": "PT30M", "label_horizon": "PT5M"},
            "cost_scenarios": ["base"],
        }
    )
    path = tmp_path / "bt.yaml"
    path.write_text(text)
    cfg = load_backtest_config(path)
    assert cfg.simulation.initial_cash == 5000.0
    assert isinstance(cfg.plan, WalkForwardPlan) and cfg.plan.effective_purge == 5 * M
    with pytest.raises(Exception):  # noqa: B017 -- any validation failure
        load_backtest_config(path.write_text(text.replace("initial_cash", "initial_cache")) or path)


def test_compare_metrics_classifies_agreement(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    results = run_sensitivity(
        config(dataset_id=ingested.dataset_id, cost_scenarios=("base", "free")),
        catalog=ingested.catalog,
        store=store,
    )
    results = {k[0]: v for k, v in results.items()}
    same = compare_metrics(results["base"].aggregate["test"], results["base"].aggregate["test"])
    assert set(same.values()) <= {Agreement.EXACT}
    different = compare_metrics(
        results["base"].aggregate["test"], results["free"].aggregate["test"]
    )
    assert different["total_costs"] is Agreement.DIFFERENT


def test_every_run_carries_its_known_limitations(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    """DEVELOPMENT_PLAN.md Stage 5: survivor-biased universes and assumed spreads are
    named on the run, not left to the reader."""
    results = run_sensitivity(
        config(dataset_id=ingested.dataset_id, cost_scenarios=("base", "free")),
        catalog=ingested.catalog,
        store=store,
    )
    results = {k[0]: v for k, v in results.items()}
    base_codes = {w.code for w in results["base"].warnings}
    assert {"static_universe", "assumed_spread"} <= base_codes
    free_codes = {w.code for w in results["free"].warnings}
    assert "zero_spread_assumed" in free_codes and "assumed_spread" not in free_codes
    assert all(w.occurrences >= 1 for w in results["base"].warnings)


def test_runs_log_with_their_run_id(ingested: IngestedFixture, store: LocalArtifactStore) -> None:
    import io
    import json
    import logging

    from qresearch.logging import configure

    stream = io.StringIO()
    configure(logging.INFO, json_lines=True, stream=stream)
    try:
        result = run_backtest(
            config(dataset_id=ingested.dataset_id), catalog=ingested.catalog, store=store
        )
    finally:
        configure(logging.WARNING)
    lines = [json.loads(ln) for ln in stream.getvalue().splitlines()]
    messages = [ln["message"] for ln in lines]
    assert "run started" in messages and "run complete" in messages
    assert all(
        ln["run_id"] == result.run_id for ln in lines if ln["message"].startswith(("run ", "fold "))
    )
    assert any(ln["message"] == "fold complete" and ln["role"] == "test" for ln in lines)


# -- fill-rule bracket -------------------------------------------------------------------------


def test_the_default_brackets_the_fill_assumption() -> None:
    cfg = BacktestConfig(
        dataset_id="x",
        strategy=StrategyRef(kind="buy_and_hold"),
        plan=WalkForwardPlan(train=H, test=H, purge=M),
    )
    assert cfg.fill_rules == (FillRule.OPEN_OF_CURRENT_BAR, FillRule.NEXT_OPEN_AFTER_ELIGIBILITY)
    assert cfg.fill_rules[0] is FillRule.OPEN_OF_CURRENT_BAR, "the headline rule leads"
    assert ExecutionConfig().fill_rule is FillRule.OPEN_OF_CURRENT_BAR, (
        "the default is the textbook rule"
    )


def test_fill_rules_are_separate_runs_and_the_optimistic_one_fills_earlier(
    ingested: IngestedFixture, store: LocalArtifactStore
) -> None:
    cfg = config(
        dataset_id=ingested.dataset_id,
        cost_scenarios=("base",),
        fill_rules=(FillRule.NEXT_OPEN_AFTER_ELIGIBILITY, FillRule.OPEN_OF_CURRENT_BAR),
    )
    results = run_sensitivity(cfg, catalog=ingested.catalog, store=store)
    assert set(results) == {
        ("base", FillRule.NEXT_OPEN_AFTER_ELIGIBILITY),
        ("base", FillRule.OPEN_OF_CURRENT_BAR),
    }
    conservative = results[("base", FillRule.NEXT_OPEN_AFTER_ELIGIBILITY)]
    optimistic = results[("base", FillRule.OPEN_OF_CURRENT_BAR)]
    assert conservative.run_id != optimistic.run_id
    assert "optimistic_fill_rule" in {w.code for w in optimistic.warnings}
    assert "optimistic_fill_rule" not in {w.code for w in conservative.warnings}

    # Same first decision, earlier fill under the optimistic rule.
    c_orders = store.load_frame(conservative.run_id, "orders").sort("signal_at")
    o_orders = store.load_frame(optimistic.run_id, "orders").sort("signal_at")
    c_fills = store.load_frame(conservative.run_id, "fills").sort("fill_at")
    o_fills = store.load_frame(optimistic.run_id, "fills").sort("fill_at")
    assert c_orders.item(0, "signal_at") == o_orders.item(0, "signal_at")
    assert o_fills.item(0, "fill_at") < c_fills.item(0, "fill_at")

    view = compare(store, [conservative.run_id, optimistic.run_id], role="test")
    assert set(view.get_column("fill_rule").to_list()) == {r.value for r in FillRule}


def test_backtest_config_fill_rules_load_from_yaml(
    tmp_path: Path, ingested: IngestedFixture
) -> None:
    path = tmp_path / "bt.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": ingested.dataset_id,
                "strategy": {"kind": "buy_and_hold", "params": {"weights": {"CRYPTO:BTCUSD": 0.5}}},
                "plan": {
                    "kind": "rolling",
                    "train": "PT1H",
                    "test": "PT30M",
                    "label_horizon": "PT5M",
                },
                "fill_rules": ["open_of_current_bar"],
            }
        )
    )
    assert load_backtest_config(path).fill_rules == (FillRule.OPEN_OF_CURRENT_BAR,)
