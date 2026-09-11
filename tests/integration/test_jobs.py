"""Job runner: queueing, progress, cancellation, crash recovery.

Most tests drive a trivial subprocess rather than a real backtest — the runner's job is
supervision, and a ten-second engine run would test the engine instead. One test runs the
real CLI end to end, and one asserts the property the whole design exists for: the command
a job runs is a command you could run yourself.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.conftest import IngestedFixture

from qresearch.jobs.contracts import JobKind, JobState
from qresearch.jobs.runner import JobRunner, cli_command
from qresearch.jobs.store import JobNotFoundError, JobStore

TIMEOUT = 30.0


def python_command(*code: str) -> list[str]:
    return [sys.executable, "-c", "; ".join(code)]


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "runs")


@pytest.fixture
def runner(store: JobStore) -> Iterator[JobRunner]:
    runner = JobRunner(store, max_concurrent=1)
    runner.start()
    yield runner
    runner.stop(timeout=TIMEOUT)


def wait_for(store: JobStore, job_id: str, *, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if store.get(job_id).state.terminal:
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: state={store.get(job_id).state}")


# -- lifecycle ---------------------------------------------------------------------------


def test_a_successful_job_runs_and_records_its_outcome(runner: JobRunner, store: JobStore) -> None:
    job = runner.submit(
        kind=JobKind.BACKTEST, command=lambda _: python_command("print('hello')"), label="smoke"
    )
    assert job.state is JobState.QUEUED
    wait_for(store, job.job_id)

    final = store.get(job.job_id)
    assert final.state is JobState.SUCCEEDED
    assert final.exit_code == 0 and final.error is None
    assert final.label == "smoke"
    assert final.started_at is not None and final.finished_at is not None
    assert final.duration is not None and final.duration >= dt.timedelta(0)
    assert "hello" in store.output_path(job.job_id).read_text()


def test_a_failing_job_records_the_exit_code_and_output(runner: JobRunner, store: JobStore) -> None:
    job = runner.submit(
        kind=JobKind.BACKTEST,
        command=lambda _: python_command("import sys", "print('boom on stderr')", "sys.exit(3)"),
    )
    wait_for(store, job.job_id)
    final = store.get(job.job_id)
    assert final.state is JobState.FAILED
    assert final.exit_code == 3
    assert final.error is not None and "boom" in final.error


def test_the_config_is_written_where_the_command_expects_it(
    runner: JobRunner, store: JobStore
) -> None:
    text = "dataset_id: ds_example\nlabel: from a job\n"
    job = runner.submit(
        kind=JobKind.BACKTEST,
        command=lambda path: python_command(f"print(open({str(path)!r}).read())"),
        config_text=text,
    )
    wait_for(store, job.job_id)
    assert store.get(job.job_id).state is JobState.SUCCEEDED
    assert store.config_path(job.job_id).read_text() == text
    assert "from a job" in store.output_path(job.job_id).read_text()


# -- progress ----------------------------------------------------------------------------


def test_structured_output_becomes_progress(runner: JobRunner, store: JobStore) -> None:
    """The runner reads the CLI's --json-logs stream; no engine callback needed."""
    events = [
        {"message": "run started", "run_id": "run_abc"},
        {"message": "fold complete", "run_id": "run_abc", "fold": 0, "role": "test"},
        {"message": "fold complete", "run_id": "run_abc", "fold": 1, "role": "test"},
        {"message": "run complete", "run_id": "run_abc"},
    ]
    code = "; ".join(f"print({json.dumps(json.dumps(e))})" for e in events)
    job = runner.submit(kind=JobKind.BACKTEST, command=lambda _: [sys.executable, "-c", code])
    wait_for(store, job.job_id)

    final = store.get(job.job_id)
    assert final.state is JobState.SUCCEEDED
    assert final.folds_complete == 2
    assert final.run_ids == ("run_abc",)
    assert final.message == "run complete"
    assert len(store.log_lines(job.job_id)) == 4


def test_unstructured_output_is_kept_but_not_parsed(runner: JobRunner, store: JobStore) -> None:
    job = runner.submit(
        kind=JobKind.BACKTEST,
        command=lambda _: python_command("print('plain text')", "print('{not json')"),
    )
    wait_for(store, job.job_id)
    assert store.get(job.job_id).state is JobState.SUCCEEDED
    assert store.log_lines(job.job_id) == []
    assert "plain text" in store.output_path(job.job_id).read_text()


# -- queueing and cancellation -------------------------------------------------------------


def test_jobs_queue_rather_than_running_at_once(runner: JobRunner, store: JobStore) -> None:
    """Concurrency is bounded so a sweep cannot fork-bomb the machine."""
    code = "import time; print('start'); time.sleep(0.6); print('end')"
    jobs = [
        runner.submit(kind=JobKind.BACKTEST, command=lambda _: [sys.executable, "-c", code])
        for _ in range(2)
    ]
    time.sleep(0.3)
    states = [store.get(j.job_id).state for j in jobs]
    assert states.count(JobState.RUNNING) <= 1, f"more than one ran at once: {states}"
    for job in jobs:
        wait_for(store, job.job_id)
    assert all(store.get(j.job_id).state is JobState.SUCCEEDED for j in jobs)


def test_cancelling_a_running_job_kills_the_process(runner: JobRunner, store: JobStore) -> None:
    code = "import time; print('started'); time.sleep(60)"
    job = runner.submit(kind=JobKind.BACKTEST, command=lambda _: [sys.executable, "-c", code])
    deadline = time.monotonic() + TIMEOUT
    while store.get(job.job_id).state is not JobState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.05)
    assert runner.cancel(job.job_id) is True
    wait_for(store, job.job_id)

    final = store.get(job.job_id)
    assert final.state is JobState.CANCELLED
    assert final.pid is not None
    from qresearch.jobs.runner import _process_alive

    assert not _process_alive(final.pid), "the child process outlived its cancellation"


def test_cancelling_a_queued_job_stops_it_starting(runner: JobRunner, store: JobStore) -> None:
    blocker = runner.submit(
        kind=JobKind.BACKTEST,
        command=lambda _: [sys.executable, "-c", "import time; time.sleep(0.8)"],
    )
    queued = runner.submit(
        kind=JobKind.BACKTEST, command=lambda _: python_command("print('should not run')")
    )
    assert runner.cancel(queued.job_id) is True
    wait_for(store, blocker.job_id)
    wait_for(store, queued.job_id)
    assert store.get(queued.job_id).state is JobState.CANCELLED
    assert (
        not store.output_path(queued.job_id).exists()
        or "should not run" not in store.output_path(queued.job_id).read_text()
    )


def test_cancelling_a_finished_job_is_a_no_op(runner: JobRunner, store: JobStore) -> None:
    job = runner.submit(kind=JobKind.BACKTEST, command=lambda _: python_command("pass"))
    wait_for(store, job.job_id)
    assert runner.cancel(job.job_id) is False
    assert store.get(job.job_id).state is JobState.SUCCEEDED


# -- crash recovery --------------------------------------------------------------------------


def test_a_job_orphaned_by_a_dead_server_is_reaped(store: JobStore) -> None:
    """A record claiming to run whose process is gone is the one state a reader cannot
    interpret, so startup resolves it instead of leaving it ambiguous."""
    orphan = store.create(kind=JobKind.BACKTEST, command=("sleep", "1000"))
    store.save(orphan.transition(JobState.RUNNING, pid=2**22, started_at=orphan.created_at))

    JobRunner(store).reap_stale()
    reaped = store.get(orphan.job_id)
    assert reaped.state is JobState.FAILED
    assert reaped.error is not None and "gone" in reaped.error


def test_reaping_leaves_terminal_jobs_alone(store: JobStore) -> None:
    done = store.create(kind=JobKind.BACKTEST, command=("true",))
    store.save(done.transition(JobState.SUCCEEDED, exit_code=0))
    JobRunner(store).reap_stale()
    assert store.get(done.job_id).state is JobState.SUCCEEDED


# -- store ------------------------------------------------------------------------------------


def test_records_round_trip_and_list_newest_first(store: JobStore) -> None:
    first = store.create(kind=JobKind.INGEST, command=("a",), label="older")
    time.sleep(0.01)
    second = store.create(kind=JobKind.BACKTEST, command=("b",), label="newer")
    assert [r.job_id for r in store.list_jobs()] == [second.job_id, first.job_id]
    assert store.get(first.job_id) == first
    assert store.exists(first.job_id) and not store.exists("job_nope")
    with pytest.raises(JobNotFoundError):
        store.get("job_nope")


def test_a_half_written_record_does_not_break_listing(store: JobStore) -> None:
    good = store.create(kind=JobKind.BACKTEST, command=("a",))
    broken = store.directory("job_broken")
    broken.mkdir(parents=True)
    (broken / "job.json").write_text("{not json")
    assert [r.job_id for r in store.list_jobs()] == [good.job_id]


# -- the reason this design exists ---------------------------------------------------------------


def test_a_job_runs_a_command_you_could_run_yourself(
    runner: JobRunner, store: JobStore, ingested: IngestedFixture, tmp_path: Path
) -> None:
    """The point of spawning the CLI (D46): the recorded command is reproducible by hand,
    and the run it produces is a normal run in a normal store."""
    import yaml

    runs_root = tmp_path / "runs"
    config = {
        "dataset_id": ingested.dataset_id,
        "strategy": {"kind": "buy_and_hold", "params": {"weights": {"CRYPTO:BTCUSD": 0.5}}},
        "plan": {"kind": "rolling", "train": "PT1H", "test": "PT30M", "label_horizon": "PT1M"},
        "cost_scenarios": ["base"],
        "fill_rules": ["open_of_current_bar"],
    }
    job = runner.submit(
        kind=JobKind.BACKTEST,
        config_text=yaml.safe_dump(config),
        command=lambda path: cli_command(
            "--verbose",
            "--json-logs",
            "backtest",
            "run",
            "-c",
            str(path),
            "--root",
            str(ingested.catalog.root),
            "--runs",
            str(runs_root),
        ),
        label="end to end",
    )
    wait_for(store, job.job_id, timeout=180.0)

    final = store.get(job.job_id)
    assert final.state is JobState.SUCCEEDED, final.error
    assert final.run_ids, "the run id was captured from the log stream"
    assert final.folds_complete > 0, "per-fold progress was observed"

    # The recorded command is literally runnable, and the run is a normal run.
    assert final.command[:3] == (sys.executable, "-m", "qresearch.cli")
    from qresearch.artifacts.local import LocalArtifactStore

    produced = LocalArtifactStore(runs_root)
    assert set(final.run_ids) <= set(produced.list_runs())
    spec = produced.load_spec(final.run_ids[0])
    assert spec.dataset_id == ingested.dataset_id
