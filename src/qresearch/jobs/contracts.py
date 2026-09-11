"""Job records.

A job is one invocation of the CLI. The record keeps the *exact command*, so a job is
always answerable: "what did this actually run, and could I run it myself?"
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum

from pydantic import Field

from qresearch.config import FrozenModel
from qresearch.time import UtcDatetime


class JobKind(StrEnum):
    BACKTEST = "backtest"
    INGEST = "ingest"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class JobRecord(FrozenModel):
    """One submitted job. Immutable; transitions produce a new record."""

    job_id: str = Field(min_length=1)
    kind: JobKind
    state: JobState
    command: tuple[str, ...] = Field(min_length=1)
    """Exactly what was executed. Copy it into a terminal and you get the same result."""

    label: str | None = None
    created_at: UtcDatetime
    started_at: UtcDatetime | None = None
    finished_at: UtcDatetime | None = None
    pid: int | None = None
    """Recorded while running so a job orphaned by a server crash can be reaped."""

    exit_code: int | None = None
    run_ids: tuple[str, ...] = ()
    folds_complete: int = Field(default=0, ge=0)
    message: str | None = None
    """Latest progress line, for display."""

    error: str | None = None

    @property
    def duration(self) -> _dt.timedelta | None:
        if self.started_at is None:
            return None
        return (self.finished_at or self.started_at) - self.started_at

    def transition(self, state: JobState, **changes: object) -> JobRecord:
        return self.model_copy(update={"state": state, **changes})
