"""The backtest launcher: form-generated configs that the CLI can reproduce.

The acceptance criterion from ``docs/workbench_plan.md`` Stage 7 is that a run launched
from the browser is indistinguishable from the same config run by hand. These tests check
that end to end, plus the validation that stops a doomed job being queued at all.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml
from tests.conftest import IngestedFixture

pytest.importorskip("fastapi", reason="needs the 'ui' extra")
pytest.importorskip("httpx2", reason="starlette's TestClient needs httpx2")

from fastapi.testclient import TestClient

from qresearch.application.config import load_backtest_config
from qresearch.artifacts.local import LocalArtifactStore
from qresearch.introspect import component_catalog
from qresearch.jobs import JobStore
from qresearch.ui.app import create_app
from qresearch.ui.launch import DEFAULT_CONFIG, LaunchError, config_to_yaml

TIMEOUT = 240.0


@pytest.fixture
def client(ingested: IngestedFixture, tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "runs", ingested.catalog.root))


@pytest.fixture
def config(ingested: IngestedFixture) -> dict[str, object]:
    return {
        **DEFAULT_CONFIG,
        "dataset_id": ingested.dataset_id,
        "plan": {"kind": "rolling", "train": "PT1H", "test": "PT30M", "label_horizon": "PT1M"},
        "cost_scenarios": ["base"],
        "fill_rules": ["open_of_current_bar"],
        "label": "from the form",
    }


# -- introspection ----------------------------------------------------------------------


def test_the_form_is_generated_from_the_registries(client: TestClient) -> None:
    """A component added in code appears in the UI with no UI change."""
    payload = client.get("/api/components").json()
    catalog = component_catalog()
    assert len(payload["features"]) == len(catalog.features)
    kinds = {f["kind"] for f in payload["features"]}
    assert {"lagged_return", "rolling_volatility", "minutes_since_open"} <= kinds

    lagged = next(f for f in payload["features"] if f["kind"] == "lagged_return")
    params = {p["name"]: p for p in lagged["params"]}
    assert params["lag"]["type"] == "int" and params["lag"]["default"] == 1
    assert params["kind"]["type"] == "choice" and params["kind"]["choices"] == ["simple", "log"]
    assert params["require_contiguous"]["type"] == "bool"


def test_every_registered_parameter_has_a_usable_control(client: TestClient) -> None:
    """An unrecognised annotation is reported rather than guessed at; none should exist."""
    payload = client.get("/api/components").json()
    unsupported = [
        (spec["kind"], param["name"], param["annotation"])
        for group in ("features", "transforms", "strategies")
        for spec in payload[group]
        for param in spec["params"]
        if param["type"] == "unsupported"
    ]
    assert unsupported == []


def test_required_parameters_are_marked(client: TestClient) -> None:
    strategies = {s["kind"]: s for s in client.get("/api/components").json()["strategies"]}
    required = {p["name"] for p in strategies["lagged_signal"]["params"] if p["required"]}
    assert required == {"feature"}, "weight and threshold have defaults"


def test_datasets_are_listed_with_what_a_picker_needs(
    client: TestClient, ingested: IngestedFixture
) -> None:
    datasets = client.get("/api/datasets").json()
    assert [d["dataset_id"] for d in datasets] == [ingested.dataset_id]
    entry = datasets[0]
    assert entry["bar_size"] == "1m" and entry["row_count"] > 0
    assert len(entry["instrument_ids"]) == 2
    assert entry["warning_count"] >= 0


# -- preview ----------------------------------------------------------------------------


def test_preview_expands_folds_without_running_anything(
    client: TestClient, config: dict[str, object], tmp_path: Path
) -> None:
    response = client.post("/api/preview", json={"config": config})
    assert response.status_code == 200
    payload = response.json()
    assert payload["estimate"]["folds_per_run"] > 0
    assert payload["estimate"]["runs"] == 1
    assert "<svg" in payload["html"]
    assert not LocalArtifactStore(tmp_path / "runs").list_runs(), "preview must not execute"
    assert not JobStore(tmp_path / "runs").list_jobs(), "preview must not queue"


def test_preview_reports_a_plan_that_cannot_fit(
    client: TestClient, config: dict[str, object]
) -> None:
    """Catching this costs a second here instead of minutes after launching."""
    doomed = {
        **config,
        "plan": {"kind": "rolling", "train": "P30D", "test": "PT1H", "purge": "PT1M"},
    }
    response = client.post("/api/preview", json={"config": doomed})
    assert response.status_code == 422
    assert "no fold fits" in response.json()["error"]


@pytest.mark.parametrize(
    ("name", "patch", "expected_field"),
    [
        ("unknown strategy", {"strategy": {"kind": "nope"}}, "strategy"),
        ("unknown feature", {"features": [{"kind": "moving_average", "params": {}}]}, "features.0"),
        (
            "unknown parameter",
            {"features": [{"kind": "lagged_return", "params": {"lags": 1}}]},
            "features.0",
        ),
        (
            "invalid value",
            {"features": [{"kind": "lagged_return", "params": {"lag": 0}}]},
            "features.0",
        ),
    ],
)
def test_bad_components_are_refused_before_queueing(
    client: TestClient,
    config: dict[str, object],
    name: str,
    patch: dict[str, object],
    expected_field: str,
) -> None:
    """Pydantic checks shape but not vocabulary: ``{"kind": "nope"}`` is a valid
    StrategyRef, and without this the job would fail minutes in."""
    response = client.post("/api/preview", json={"config": {**config, **patch}})
    assert response.status_code == 422, name
    fields = {e["field"] for e in response.json()["errors"]}
    assert expected_field in fields, (name, response.json())


def test_a_missing_dataset_is_a_field_error(client: TestClient, config: dict[str, object]) -> None:
    without = {k: v for k, v in config.items() if k != "dataset_id"}
    response = client.post("/api/preview", json={"config": without})
    assert response.status_code == 422
    assert any(e["field"] == "dataset_id" for e in response.json()["errors"])


# -- launch -----------------------------------------------------------------------------


def test_the_config_the_form_writes_is_the_config_the_cli_parses(
    config: dict[str, object], tmp_path: Path
) -> None:
    text = config_to_yaml(config)
    path = tmp_path / "round_trip.yaml"
    path.write_text(text)
    loaded = load_backtest_config(path)
    assert loaded.dataset_id == config["dataset_id"]
    assert [f.kind for f in loaded.features] == [f["kind"] for f in config["features"]]  # type: ignore[index]
    assert loaded.label == "from the form"
    assert yaml.safe_load(text)["plan"]["kind"] == "rolling"


def test_launching_queues_a_job_and_produces_a_normal_run(
    client: TestClient, config: dict[str, object], tmp_path: Path
) -> None:
    response = client.post("/api/backtests", json={"config": config, "label": "from the form"})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert "backtest" in response.json()["command"]

    jobs = JobStore(tmp_path / "runs")
    deadline = time.monotonic() + TIMEOUT
    while not jobs.get(job_id).state.terminal and time.monotonic() < deadline:
        time.sleep(0.2)
    job = jobs.get(job_id)
    assert job.state.value == "succeeded", job.error
    assert job.folds_complete > 0 and job.run_ids

    store = LocalArtifactStore(tmp_path / "runs")
    assert set(job.run_ids) <= set(store.list_runs())
    spec = store.load_spec(job.run_ids[0])
    assert spec.label == "from the form"
    assert spec.dataset_id == config["dataset_id"]
    assert spec.simulation.execution.fill_rule.value == "open_of_current_bar"


def test_the_job_config_is_the_file_a_human_would_write(
    client: TestClient, config: dict[str, object], tmp_path: Path
) -> None:
    """Stage 7's acceptance criterion: the browser path and the YAML path are one path."""
    job_id = client.post("/api/backtests", json={"config": config}).json()["job_id"]
    jobs = JobStore(tmp_path / "runs")
    written = jobs.config_path(job_id).read_text()
    assert load_backtest_config(jobs.config_path(job_id)).dataset_id == config["dataset_id"]
    assert written == config_to_yaml(config)


def test_launching_an_invalid_config_queues_nothing(
    client: TestClient, config: dict[str, object], tmp_path: Path
) -> None:
    response = client.post(
        "/api/backtests", json={"config": {**config, "strategy": {"kind": "nope"}}}
    )
    assert response.status_code == 422
    assert JobStore(tmp_path / "runs").list_jobs() == []


def test_the_launcher_page_and_its_assets_are_served(client: TestClient) -> None:
    page = client.get("/new")
    assert page.status_code == 200
    assert "const DEFAULTS=" in page.text
    assert "/static/launch.js" in page.text
    assert client.get("/static/launch.js").status_code == 200
    assert client.get("/static/ui.css").status_code == 200


def test_defaults_are_json_serialisable_and_parse_as_durations() -> None:
    """The defaults are rendered into form fields and posted back, so they must already
    be in the wire format the parser accepts -- a timedelta would render as '6:00:00'."""
    json.dumps(DEFAULT_CONFIG)
    assert DEFAULT_CONFIG["plan"]["train"] == "PT6H"  # type: ignore[index]
    assert DEFAULT_CONFIG["fill_rules"] == ["open_of_current_bar", "next_open_after_eligibility"]


def test_launch_error_carries_field_detail() -> None:
    with pytest.raises(LaunchError) as caught:
        config_to_yaml({"strategy": {"kind": "buy_and_hold"}})
    assert caught.value.errors
