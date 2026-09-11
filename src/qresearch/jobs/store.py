"""Filesystem job store.

One directory per job under ``<runs_root>/.jobs/``, holding the record, the config that
was run, and the captured output. Same append-only, atomically-published pattern as run
artifacts: a reader never sees a half-written record.

Lives beside the runs it produces rather than in a separate database so that a job, its
config, and its results are one thing you can archive or delete together.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from qresearch.jobs.contracts import JobKind, JobRecord, JobState
from qresearch.time import now_utc

JOBS_DIRNAME = ".jobs"


class JobNotFoundError(LookupError):
    """No job with that id under the store root."""


class JobStore:
    def __init__(self, runs_root: Path | str) -> None:
        self.root = Path(runs_root) / JOBS_DIRNAME

    # -- paths -------------------------------------------------------------------

    def directory(self, job_id: str) -> Path:
        return self.root / job_id

    def config_path(self, job_id: str) -> Path:
        return self.directory(job_id) / "config.yaml"

    def log_path(self, job_id: str) -> Path:
        return self.directory(job_id) / "log.jsonl"

    def output_path(self, job_id: str) -> Path:
        return self.directory(job_id) / "output.txt"

    # -- records -----------------------------------------------------------------

    def create(
        self,
        *,
        kind: JobKind,
        command: tuple[str, ...],
        config_text: str | None = None,
        label: str | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        created = now_utc()
        job_id = job_id or f"job_{created:%Y%m%dT%H%M%S%f}_{os.getpid()}"
        directory = self.directory(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        if config_text is not None:
            self.config_path(job_id).write_text(config_text, encoding="utf-8")
        record = JobRecord(
            job_id=job_id,
            kind=kind,
            state=JobState.QUEUED,
            command=command,
            label=label,
            created_at=created,
        )
        self.save(record)
        return record

    def save(self, record: JobRecord) -> None:
        """Atomically publish a record: write a sibling, then rename over."""
        directory = self.directory(record.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "job.json"
        tmp = directory / "job.json.tmp"
        tmp.write_text(
            json.dumps(json.loads(record.model_dump_json()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(target)

    def get(self, job_id: str) -> JobRecord:
        path = self.directory(job_id) / "job.json"
        if not path.exists():
            raise JobNotFoundError(f"no job {job_id!r} under {self.root}")
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def exists(self, job_id: str) -> bool:
        return (self.directory(job_id) / "job.json").exists()

    def list_jobs(self, *, limit: int | None = None) -> list[JobRecord]:
        """Newest first."""
        if not self.root.exists():
            return []
        records = []
        for directory in self.root.iterdir():
            if (directory / "job.json").exists():
                try:
                    records.append(self.get(directory.name))
                except (ValueError, OSError):
                    continue  # a record mid-write or hand-edited; skip rather than fail
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit] if limit else records

    def log_lines(self, job_id: str, *, limit: int = 200) -> list[dict[str, object]]:
        path = self.log_path(job_id)
        if not path.exists():
            return []
        out: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
