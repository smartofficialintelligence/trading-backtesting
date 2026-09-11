"""Structured logging with a ``run_id`` context.

Every record carries the run id of the backtest in progress (or ``-``), set with
:func:`run_context`, so log lines from one run can be pulled out of an interleaved
stream. Output is either a human line or one JSON object per line.

Off by default: libraries should not configure logging on import. The CLI calls
:func:`configure`; tests and notebooks call it if they want output.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "qresearch_run_id", default=None
)
ROOT = "qresearch"


class RunContextFilter(logging.Filter):
    """Attach the current run id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure(
    level: int | str = logging.WARNING, *, json_lines: bool = False, stream: Any = None
) -> None:
    """Install one handler on the ``qresearch`` logger. Idempotent."""
    logger = logging.getLogger(ROOT)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.addFilter(RunContextFilter())
    handler.setFormatter(
        JsonFormatter()
        if json_lines
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s run=%(run_id)s %(message)s")
    )
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith(ROOT) else f"{ROOT}.{name}")


@contextmanager
def run_context(run_id: str | None) -> Iterator[None]:
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


def current_run_id() -> str | None:
    return _run_id.get()
