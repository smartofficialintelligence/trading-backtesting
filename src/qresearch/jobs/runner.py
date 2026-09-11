"""Runs jobs as CLI subprocesses, one at a time by default.

A worker thread supervises; the work itself happens in a separate process. The thread is
only waiting on pipe reads, so it costs nothing, and the GIL never contends with the
engine.

Progress comes free. ``qresearch --verbose --json-logs`` already emits one JSON object per
line — ``run started``, ``fold complete`` (with fold, role, fills, return), ``run
complete`` — each tagged with ``run_id``. The runner tails that stream, so a UI can show
real progress without the engine growing a callback interface it does not otherwise need.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from qresearch.jobs.contracts import JobKind, JobRecord, JobState
from qresearch.jobs.store import JobStore
from qresearch.logging import get_logger
from qresearch.time import now_utc

log = get_logger(__name__)

CommandFactory = Callable[[Path], Sequence[str]]
"""Builds the argv for a job, given the path its config was written to."""

TERMINATE_GRACE_SECONDS = 5.0


def cli_command(*args: str) -> list[str]:
    """Invoke this interpreter's qresearch CLI.

    ``sys.executable -m`` rather than the console script so a job runs under the same
    interpreter and virtualenv as the server, whatever the PATH happens to be.
    """
    return [sys.executable, "-m", "qresearch.cli", *args]


class JobRunner:
    """A bounded queue of CLI subprocesses.

    Default concurrency is 1: a backtest is CPU- and memory-hungry (the recorded baseline
    peaks around 300 MB), and a sweep submitting fifty jobs must not try to run fifty
    processes.
    """

    def __init__(self, store: JobStore, *, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.store = store
        self.max_concurrent = max_concurrent
        self._queue: queue.Queue[str] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        if self._workers:
            return
        self.reap_stale()
        for index in range(self.max_concurrent):
            worker = threading.Thread(target=self._work, name=f"qresearch-job-{index}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def stop(self, *, timeout: float = 10.0) -> None:
        """Stop accepting work and wait for the current jobs to finish."""
        self._stopping.set()
        for _ in self._workers:
            self._queue.put("")
        for worker in self._workers:
            worker.join(timeout=timeout)
        self._workers.clear()

    def reap_stale(self) -> None:
        """Mark jobs left running by a previous process as failed.

        A record claiming to be running whose process is gone is the one state a reader
        cannot interpret, so it is resolved at startup rather than left ambiguous.
        """
        for record in self.store.list_jobs():
            if record.state.terminal:
                continue
            if record.pid is not None and _process_alive(record.pid):
                continue  # genuinely still running, e.g. a second server
            self.store.save(
                record.transition(
                    JobState.FAILED,
                    finished_at=now_utc(),
                    error="interrupted: the process running this job is gone",
                )
            )

    # -- submission --------------------------------------------------------------

    def submit(
        self,
        *,
        kind: JobKind,
        command: CommandFactory,
        config_text: str | None = None,
        label: str | None = None,
    ) -> JobRecord:
        """Queue a job. Returns immediately with a ``queued`` record."""
        record = self.store.create(
            kind=kind, command=("pending",), config_text=config_text, label=label
        )
        resolved = tuple(str(part) for part in command(self.store.config_path(record.job_id)))
        record = record.model_copy(update={"command": resolved})
        self.store.save(record)
        self._queue.put(record.job_id)
        log.info("job queued", extra={"fields": {"job": record.job_id, "kind": kind.value}})
        return record

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job. Returns False if already terminal."""
        record = self.store.get(job_id)
        if record.state.terminal:
            return False
        with self._lock:
            self._cancelled.add(job_id)
            process = self._processes.get(job_id)
        if process is not None:
            _terminate(process)
        else:
            # Still queued: mark it now; the worker will skip it.
            self.store.save(
                record.transition(
                    JobState.CANCELLED, finished_at=now_utc(), message="cancelled before start"
                )
            )
        return True

    # -- worker ------------------------------------------------------------------

    def _work(self) -> None:
        while not self._stopping.is_set():
            job_id = self._queue.get()
            if not job_id:
                self._queue.task_done()
                continue
            try:
                self._run(job_id)
            except Exception as error:
                log.exception("job crashed", extra={"fields": {"job": job_id}})
                with contextlib.suppress(Exception):  # nothing left to do if this fails
                    self.store.save(
                        self.store.get(job_id).transition(
                            JobState.FAILED,
                            finished_at=now_utc(),
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        record = self.store.get(job_id)
        with self._lock:
            if job_id in self._cancelled or record.state is JobState.CANCELLED:
                self._cancelled.discard(job_id)
                return

        # Context-managed so the pipes are closed even if consuming raises: an unclosed
        # stdout leaks a file descriptor per job, which a long-lived server would feel.
        with subprocess.Popen(
            list(record.command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        ) as process:
            with self._lock:
                self._processes[job_id] = process
            record = record.transition(JobState.RUNNING, started_at=now_utc(), pid=process.pid)
            self.store.save(record)
            log.info("job started", extra={"fields": {"job": job_id, "pid": process.pid}})

            try:
                record = self._consume(record, process)
            finally:
                exit_code = process.wait()

        with self._lock:
            self._processes.pop(job_id, None)
            cancelled = job_id in self._cancelled
            self._cancelled.discard(job_id)

        if cancelled:
            final = record.transition(
                JobState.CANCELLED, finished_at=now_utc(), exit_code=exit_code
            )
        elif exit_code == 0:
            final = record.transition(
                JobState.SUCCEEDED, finished_at=now_utc(), exit_code=exit_code
            )
        else:
            final = record.transition(
                JobState.FAILED,
                finished_at=now_utc(),
                exit_code=exit_code,
                error=_tail(self.store.output_path(job_id)) or f"exited with code {exit_code}",
            )
        self.store.save(final)
        log.info("job finished", extra={"fields": {"job": job_id, "state": final.state.value}})

    def _consume(self, record: JobRecord, process: subprocess.Popen[str]) -> JobRecord:
        """Tail the child's output, persisting raw text and parsed progress."""
        log_path = self.store.log_path(record.job_id)
        output_path = self.store.output_path(record.job_id)
        run_ids: list[str] = list(record.run_ids)
        folds = record.folds_complete
        message = record.message
        last_saved = 0.0

        assert process.stdout is not None
        with (
            output_path.open("w", encoding="utf-8") as raw,
            log_path.open("w", encoding="utf-8") as structured,
        ):
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                event = _parse(line)
                if event is None:
                    continue
                structured.write(json.dumps(event, sort_keys=True) + "\n")
                structured.flush()
                run_id = event.get("run_id")
                if isinstance(run_id, str) and run_id not in ("-", "") and run_id not in run_ids:
                    run_ids.append(run_id)
                if event.get("message") == "fold complete":
                    folds += 1
                if isinstance(event.get("message"), str):
                    message = str(event["message"])
                # Persist progress at most a few times a second: a UI polls, and a run
                # emits far more lines than anyone needs to see written to disk.
                now = now_utc().timestamp()
                if now - last_saved > 0.4:
                    last_saved = now
                    record = record.model_copy(
                        update={
                            "run_ids": tuple(run_ids),
                            "folds_complete": folds,
                            "message": message,
                        }
                    )
                    self.store.save(record)
        return record.model_copy(
            update={"run_ids": tuple(run_ids), "folds_complete": folds, "message": message}
        )


def _parse(line: str) -> dict[str, object] | None:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _tail(path: Path, *, lines: int = 12) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return "\n".join(text.splitlines()[-lines:]) if text else None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _terminate(process: subprocess.Popen[str]) -> None:
    """Ask politely, then insist."""
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
    except OSError:
        pass
