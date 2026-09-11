"""The local UI, exercised through the app rather than a browser.

Read-only by design today. The test that matters most is the last one: the UI must not
become a second way to produce results, or the reproducibility guarantees stop meaning
anything.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from tests.conftest import IngestedFixture

from qresearch.application.run_backtest import BacktestConfig, run_backtest
from qresearch.artifacts.contracts import FeatureRef, StrategyRef
from qresearch.artifacts.local import LocalArtifactStore
from qresearch.research.walk_forward import WalkForwardPlan
from qresearch.simulation.engine import SimulationConfig
from qresearch.simulation.execution import ExecutionConfig, FillRule

pytest.importorskip("fastapi", reason="needs the 'ui' extra")
pytest.importorskip("httpx2", reason="starlette's TestClient needs httpx2")

from fastapi.testclient import TestClient

from qresearch.ui.app import create_app

H, M = dt.timedelta(hours=1), dt.timedelta(minutes=1)


@pytest.fixture
def populated(ingested: IngestedFixture, tmp_path: Path) -> tuple[LocalArtifactStore, list[str]]:
    """Two real runs differing only in fill rule — the bracket, as the UI will show it."""
    store = LocalArtifactStore(tmp_path / "runs")
    config = BacktestConfig(
        dataset_id=ingested.dataset_id,
        features=(FeatureRef(kind="lagged_return", params={"lag": 1}),),
        strategy=StrategyRef(kind="lagged_signal", params={"feature": "ret_1", "weight": 0.4}),
        simulation=SimulationConfig(
            initial_cash=10_000.0, execution=ExecutionConfig(liquidity_lookback_bars=5)
        ),
        plan=WalkForwardPlan(train=H, test=30 * M, purge=2 * M),
        cost_scenarios=("base",),
        label="ui fixture",
    )
    ids = [
        run_backtest(
            config, catalog=ingested.catalog, store=store, cost_scenario="base", fill_rule=rule
        ).run_id
        for rule in (FillRule.OPEN_OF_CURRENT_BAR, FillRule.NEXT_OPEN_AFTER_ELIGIBILITY)
    ]
    return store, ids


@pytest.fixture
def client(populated: tuple[LocalArtifactStore, list[str]]) -> TestClient:
    store, _ = populated
    return TestClient(create_app(store.root, "data"))


def test_health_reports_the_store(client: TestClient, populated) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok" and payload["runs"] == 2


def test_the_index_lists_every_run_with_its_assumptions(client: TestClient, populated) -> None:
    _, ids = populated
    page = client.get("/")
    assert page.status_code == 200
    for run_id in ids:
        assert run_id in page.text
    # Assumptions are on the row, not hidden behind a click.
    assert "open_of_current_bar" in page.text and "next_open_after_eligibility" in page.text
    assert page.text.count("data-run=") == 2


def test_an_empty_store_explains_what_to_do(tmp_path: Path) -> None:
    page = TestClient(create_app(tmp_path / "empty", "data")).get("/")
    assert page.status_code == 200
    assert "No runs yet" in page.text and "qresearch backtest run" in page.text


def test_run_detail_serves_the_archival_report(client: TestClient, populated) -> None:
    """One renderer for both, so the browser view and the stored file cannot drift."""
    store, ids = populated
    response = client.get(f"/runs/{ids[0]}")
    assert response.status_code == 200
    assert "What this run assumed" in response.text
    assert response.text == (store.path_for(ids[0]) / "report.html").read_text()


def test_run_detail_regenerates_a_missing_report(client: TestClient, populated) -> None:
    """Reports are generated on demand, not by the library run: a run made through
    ``run_backtest`` has no report until something asks for one, and a deleted report
    comes back rather than 404ing."""
    store, ids = populated
    report = store.path_for(ids[0]) / "report.html"
    assert not report.exists(), "the library run does not render a presentation artifact"

    assert client.get(f"/runs/{ids[0]}").status_code == 200
    assert report.exists(), "first view generates it"

    report.unlink()
    assert client.get(f"/runs/{ids[0]}").status_code == 200
    assert report.exists(), "and it comes back if deleted"


def test_unknown_run_is_a_404(client: TestClient, populated) -> None:
    assert client.get("/runs/run_nope").status_code == 404


def test_compare_overlays_curves_and_leads_with_assumptions(client: TestClient, populated) -> None:
    _, ids = populated
    page = client.get("/compare", params={"runs": ",".join(ids), "role": "test"})
    assert page.status_code == 200
    assert "<svg" in page.text
    assert page.text.index("Assumptions") < page.text.index("Performance"), "assumptions first"
    assert "open_of_current_bar" in page.text and "next_open_after_eligibility" in page.text


def test_compare_with_nothing_selected_is_not_an_error(client: TestClient, populated) -> None:
    page = client.get("/compare", params={"runs": ""})
    assert page.status_code == 200 and "Nothing selected" in page.text


def test_compare_ignores_unknown_runs(client: TestClient, populated) -> None:
    _, ids = populated
    page = client.get("/compare", params={"runs": f"{ids[0]},run_nope"})
    assert page.status_code == 200 and ids[0] in page.text


def test_ledger_endpoint_serves_known_ledgers_only(client: TestClient, populated) -> None:
    _, ids = populated
    ok = client.get(f"/api/runs/{ids[0]}/ledger/trades")
    assert ok.status_code == 200 and len(json.loads(ok.text)) > 0
    assert client.get(f"/api/runs/{ids[0]}/ledger/run_spec").status_code == 400
    assert client.get(f"/api/runs/{ids[0]}/ledger/../../etc/passwd").status_code in (400, 404)
    assert client.get("/api/runs/run_nope/ledger/fills").status_code == 404


def test_the_ui_reads_the_same_run_table_the_cli_does(client: TestClient, populated) -> None:
    """No UI-specific view of a run: one source, so the browser and CLI cannot disagree."""
    from qresearch.research.experiments import run_table

    store, _ = populated
    api = json.loads(client.get("/api/runs").json())
    assert len(api) == run_table(store).height
    assert {r["run_id"] for r in api} == set(run_table(store).get_column("run_id").to_list())


def test_the_ui_never_writes_to_the_run_store_itself(client: TestClient, populated) -> None:
    """Replaces the earlier GET/HEAD-only assertion, which stopped holding when job
    control arrived (see docs/workbench_plan.md Stage 7).

    The guarantee that actually matters is not "no mutation" but **"no mutation of
    results"**: the run store is written only by the CLI subprocess, so anything the UI
    produces is reproducible by hand. Browsing must not create, alter, or delete a run.
    """
    store, ids = populated
    before = set(store.list_runs())
    digests = {run_id: store.load_result(run_id).economic_digest for run_id in before}

    for path in ("/", "/jobs", "/compare", f"/runs/{ids[0]}", "/api/runs", "/api/jobs"):
        assert client.get(path).status_code == 200, path

    assert set(store.list_runs()) == before, "browsing changed the set of runs"
    assert {r: store.load_result(r).economic_digest for r in before} == digests


def test_every_mutating_route_goes_through_the_cli(client: TestClient, populated) -> None:
    """Each non-GET route must either queue work for the CLI subprocess or control a job.

    Asserted by equality so a new mutating route fails this test until someone writes down
    why it is safe. The guarantee it protects (D46): the run store is written only by the
    subprocess, so anything the UI produces is reproducible by hand.

    * ``POST /api/preview`` — a pure computation that writes nothing; a POST only because
      it takes a config body, which is awkward on a GET.
    * ``POST /api/backtests`` — validates a config and queues ``qresearch backtest run``.
      It creates a *job*, never a run.
    * ``DELETE /api/jobs/{job_id}`` — stops unfinished work. Destroys nothing.
    """
    store, _ = populated
    app = create_app(store.root, "data")
    mutating = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if method not in {"GET", "HEAD", "OPTIONS"}
    }
    assert mutating == {
        ("POST", "/api/preview"),
        ("POST", "/api/backtests"),
        ("DELETE", "/api/jobs/{job_id}"),
    }, f"unexpected mutating routes: {mutating}. Route it through the CLI or justify it here."


def test_launching_creates_a_job_not_a_run(client: TestClient, populated) -> None:
    """The launch route must not write to the run store itself."""
    from qresearch.ui.launch import DEFAULT_CONFIG

    store, _ = populated
    before = set(store.list_runs())
    datasets = client.get("/api/datasets").json()
    if not datasets:
        pytest.skip("no dataset in this fixture's catalog")
    config = {**DEFAULT_CONFIG, "dataset_id": datasets[0]["dataset_id"]}
    response = client.post("/api/backtests", json={"config": config})
    assert response.status_code == 202
    assert set(store.list_runs()) == before, "the route wrote a run directly"
    command = response.json()["command"]
    assert "qresearch.cli" in " ".join(command) and "backtest" in command


def test_cancelling_an_unknown_job_is_a_404(client: TestClient, populated) -> None:
    assert client.delete("/api/jobs/job_nope").status_code == 404


def test_the_jobs_page_renders_when_empty(client: TestClient, populated) -> None:
    page = client.get("/jobs")
    assert page.status_code == 200 and "No jobs" in page.text
    assert client.get("/api/jobs").json() == []


def test_a_job_appears_on_the_jobs_page_with_its_command(client: TestClient, populated) -> None:
    """The job page shows the exact command, because being able to re-run it by hand is
    the whole point of the subprocess design."""
    import sys

    from qresearch.jobs import JobKind, JobStore
    from qresearch.jobs.runner import JobRunner

    store, _ = populated
    jobs = JobStore(store.root)
    runner = JobRunner(jobs)
    record = jobs.create(
        kind=JobKind.BACKTEST,
        command=(sys.executable, "-m", "qresearch.cli", "version"),
        label="visible",
    )
    del runner

    page = client.get("/jobs")
    assert "visible" in page.text and "queued" in page.text
    detail = client.get(f"/jobs/{record.job_id}")
    assert detail.status_code == 200
    assert "qresearch.cli" in detail.text
    assert "run this yourself" in detail.text
    assert client.get("/jobs/job_nope").status_code == 404
