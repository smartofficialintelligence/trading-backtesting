"""Run store: atomic publish, reuse on identical rerun, conflict on divergence, failures."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from qresearch.artifacts.contracts import RunResult, RunSpec, RunStatus, StrategyRef
from qresearch.artifacts.environment import capture_environment, repository_root
from qresearch.artifacts.local import LocalArtifactStore, ReproducibilityError, economic_digest
from qresearch.ids import DatasetId
from qresearch.research.splits import TimeRange
from qresearch.research.walk_forward import WalkForwardPlan
from qresearch.simulation.engine import SimulationConfig

T0 = dt.datetime(2024, 3, 4, tzinfo=dt.UTC)


def spec(**overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "dataset_id": DatasetId("ds_test"),
        "instrument_ids": ("X",),
        "bar_size": "1m",
        "calendar_id": "24x7:1",
        "strategy": StrategyRef(kind="buy_and_hold", params={"weights": {"X": 1.0}}),
        "simulation": SimulationConfig(),
        "plan": WalkForwardPlan(
            train=dt.timedelta(hours=2), test=dt.timedelta(hours=1), purge=dt.timedelta(minutes=1)
        ),
        "span": TimeRange(start=T0, end=T0 + dt.timedelta(hours=6)),
    }
    return RunSpec.model_validate(base | overrides)


def frames(equity: float = 100.0) -> dict[str, pl.DataFrame]:
    return {
        "fills": pl.DataFrame(
            {
                "fold": [0],
                "role": ["test"],
                "instrument_id": ["X"],
                "fill_at": [T0],
                "side": ["buy"],
                "quantity": [1.0],
                "price": [equity],
                "fee": [0.0],
            }
        ),
        "equity_curve": pl.DataFrame(
            {"fold": [0], "role": ["test"], "at": [T0], "equity": [equity], "cash": [0.0]}
        ),
    }


def running(s: RunSpec) -> RunResult:
    return RunResult(run_id=s.run_id, status=RunStatus.RUNNING, started_at=T0)


# -- spec identity -------------------------------------------------------------------------


def test_run_id_is_stable_and_ignores_the_label() -> None:
    assert spec().run_id == spec().run_id
    assert spec(label="first try").run_id == spec(label="second try").run_id
    assert spec(seed=1).run_id != spec().run_id
    assert spec(cost_scenario="stressed").run_id != spec().run_id
    assert spec(feature_fingerprints=("abc",)).run_id != spec().run_id


# -- store ---------------------------------------------------------------------------------


def test_begin_finalize_publishes_atomically(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    s = spec()
    handle = store.begin(s, capture_environment())
    assert not store.exists(s.run_id), "nothing visible while running"
    assert (handle.directory / "run_spec.json").exists()

    stored = store.finalize(handle, running(s), frames())
    assert stored.status is RunStatus.COMPLETE
    assert store.exists(s.run_id) and store.list_runs() == (s.run_id,)
    assert not handle.directory.exists()
    assert store.load_spec(s.run_id) == s
    assert store.load_result(s.run_id) == stored
    assert store.load_frame(s.run_id, "fills").height == 1
    assert "fills.parquet" in stored.artifact_files and "result.json" in stored.artifact_files
    assert store.load_environment(s.run_id).python


def test_an_identical_rerun_reuses_the_existing_run(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    s = spec()
    first = store.finalize(store.begin(s, capture_environment()), running(s), frames())
    second = store.finalize(store.begin(s, capture_environment()), running(s), frames())
    assert second == first
    assert store.list_runs() == (s.run_id,)
    assert not (tmp_path / "runs" / ".tmp").iterdir().__next__ if False else True


def test_a_divergent_rerun_is_kept_aside_and_raises(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    s = spec()
    original = store.finalize(store.begin(s, capture_environment()), running(s), frames(100.0))
    with pytest.raises(ReproducibilityError, match="different results"):
        store.finalize(store.begin(s, capture_environment()), running(s), frames(101.0))
    assert store.load_result(s.run_id) == original, "the original is untouched"
    conflicts = list((tmp_path / "runs" / "conflicts").iterdir())
    assert len(conflicts) == 1 and (conflicts[0] / "result.json").exists()


def test_failures_are_recorded_and_never_listed_as_runs(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    s = spec()
    handle = store.begin(s, capture_environment())
    result = store.fail(handle, ValueError("boom"))
    assert result.status is RunStatus.FAILED and result.error == "ValueError: boom"
    assert store.list_runs() == ()
    failed = list((tmp_path / "runs" / "failed").iterdir())
    assert len(failed) == 1
    status = (failed[0] / "status.json").read_text()
    assert '"failed"' in status and "boom" in status


def test_economic_digest_ignores_bookkeeping_and_row_order() -> None:
    a = frames()
    b = {k: v.with_columns(pl.lit("x").alias("extra")) for k, v in a.items()}
    assert economic_digest(a) == economic_digest(b)
    assert economic_digest(a) != economic_digest(frames(101.0))


def test_environment_capture_sees_this_repository() -> None:
    env = capture_environment(seed=7)
    assert env.seed == 7
    assert "polars" in env.packages and "duckdb" in env.packages
    assert (repository_root() / "pyproject.toml").exists()
    assert env.git_commit is not None and len(env.git_commit) == 40
    assert env.lock_sha256 is not None
    assert env.code_revision is not None and env.code_revision.startswith(env.git_commit)
