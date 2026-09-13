"""Sealed test windows: periods no development run may evaluate.

A holdout is evidence only while nobody has looked at how a strategy did in it. Once a
window's result has been seen and the strategy adjusted, the window is development data
whatever it is called. Recording the windows in a file is not enough when development
data and test windows live in the same dataset, so the orchestrator enforces them: a run
with any fold touching a sealed window is refused unless the config either skips such
folds (``exclude_sealed``, for development) or names the window (``unseal``, for the
one-time evaluation, recorded in the run spec).

Each window carries a gap on both sides. A fold touches a window if any part of its span
(warm-up through test) falls inside the window or its gaps, so no development decision is
made next to a test period and no development position is held into one. Bars inside a
window may still be read as *lookback history* by decisions after its gap, as they would
be in live trading: that uses the window's prices, not a strategy's results in it.
"""

from __future__ import annotations

import datetime as _dt
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import polars as pl
import yaml
from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.research.splits import TimeRange

REGISTRY_ENV = "QRESEARCH_SEALED_WINDOWS"
DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "configs" / "sealed_windows.yaml"


class SealedWindowError(ValueError):
    """A run would evaluate a sealed window it did not name."""


class SealedWindow(FrozenModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    range: TimeRange
    gap: _dt.timedelta = Field(ge=_dt.timedelta(0))
    note: str = ""

    @property
    def guarded(self) -> TimeRange:
        """The window plus its gap on both sides."""
        return TimeRange(start=self.range.start - self.gap, end=self.range.end + self.gap)


class SealedWindows(FrozenModel):
    windows: tuple[SealedWindow, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [w.name for w in self.windows]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"sealed window names must be unique; repeated: {duplicates}")
        return self

    def get(self, name: str) -> SealedWindow:
        for window in self.windows:
            if window.name == name:
                return window
        raise ValueError(
            f"no sealed window named {name!r}; known: {sorted(w.name for w in self.windows)}"
        )

    def touching(self, span: TimeRange) -> tuple[SealedWindow, ...]:
        return tuple(w for w in self.windows if w.guarded.overlaps(span))

    def excluded_ranges(self, *, keep: Sequence[str] = ()) -> tuple[TimeRange, ...]:
        """Guarded ranges of every window except those named in ``keep``."""
        return tuple(w.guarded for w in self.windows if w.name not in keep)

    def development_predicate(self, column: str = "bar_start") -> pl.Expr:
        """True for rows outside every window and its gaps -- for exploratory studies."""
        expr = pl.lit(True)
        for window in self.windows:
            expr = expr & ~window.guarded.predicate(column)
        return expr

    @classmethod
    def load(cls, path: Path | str) -> Self:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data, strict=False)

    @classmethod
    def default(cls) -> Self:
        """The repository registry, or ``$QRESEARCH_SEALED_WINDOWS``; empty if absent.

        Located relative to the source tree rather than the working directory, so a run
        launched from another directory -- a job, a script -- is guarded the same way.
        """
        path = Path(os.environ.get(REGISTRY_ENV, DEFAULT_REGISTRY))
        return cls.load(path) if path.exists() else cls()
