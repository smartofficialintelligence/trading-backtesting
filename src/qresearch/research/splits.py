"""Time ranges, split roles, and role assignment.

The minimal contract needed so that a fitted transform can *refuse* non-training data
(DEVELOPMENT_PLAN.md Stage 2 acceptance). Walk-forward fold generation, purging, and
embargo build on these in Stage 4.

Rows are assigned to a split by their ``available_at``, not by ``bar_start``. A split is a
period of *knowledge*: the training set is what was knowable by the training cutoff. A bar
whose interval falls in the training range but which published after the cutoff was not
knowable then, and belongs to whatever period it published in.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from enum import StrEnum
from typing import Self

import polars as pl
from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.time import UtcDatetime


class SplitRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    WARMUP = "warmup"
    """Data supplied for lookback only; never evaluated, never fitted on."""

    PURGE = "purge"
    """Removed between train and later roles so labels cannot straddle the boundary."""

    EMBARGO = "embargo"
    """Removed after test data that a fit could otherwise see through overlap."""


FITTABLE_ROLES: frozenset[SplitRole] = frozenset({SplitRole.TRAIN})
"""Roles whose data a transform may be fitted on. Deliberately a set of one."""


class TimeRange(FrozenModel):
    """Half-open UTC interval ``[start, end)``."""

    start: UtcDatetime
    end: UtcDatetime

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start >= self.end:
            raise ValueError(
                f"start {self.start} must precede end {self.end}; ranges are half-open "
                "and non-empty"
            )
        return self

    @property
    def duration(self) -> _dt.timedelta:
        return self.end - self.start

    def contains(self, moment: _dt.datetime) -> bool:
        return self.start <= moment < self.end

    def overlaps(self, other: TimeRange) -> bool:
        return self.start < other.end and other.start < self.end

    def predicate(self, column: str = "available_at") -> pl.Expr:
        return (pl.col(column) >= self.start) & (pl.col(column) < self.end)


class DataSplit(FrozenModel):
    """One role over one range, within one fold."""

    role: SplitRole
    range: TimeRange
    fold: int = Field(default=0, ge=0)

    @property
    def fittable(self) -> bool:
        return self.role in FITTABLE_ROLES


def check_disjoint(splits: Sequence[DataSplit]) -> None:
    """Raise if any two splits in the same fold overlap in time."""
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            if a.fold == b.fold and a.range.overlaps(b.range):
                raise ValueError(
                    f"fold {a.fold}: {a.role.value} [{a.range.start}, {a.range.end}) "
                    f"overlaps {b.role.value} [{b.range.start}, {b.range.end})"
                )


def select_split(
    frame: pl.DataFrame, split: DataSplit, *, on: str = "available_at"
) -> pl.DataFrame:
    """Rows of ``frame`` belonging to ``split`` by the ``on`` timestamp."""
    if on not in frame.columns:
        raise ValueError(f"frame has no {on!r} column to assign splits by")
    return frame.filter(split.range.predicate(on))


def assign_roles(
    frame: pl.DataFrame, splits: Sequence[DataSplit], *, on: str = "available_at"
) -> pl.DataFrame:
    """Add ``split_role`` and ``fold`` columns; rows outside every split are dropped.

    Splits in the same fold must be disjoint. A row may appear once per fold it falls in,
    so the result can be longer than the input when folds overlap (as walk-forward folds
    do) -- callers filter by ``fold``.
    """
    if not splits:
        raise ValueError("no splits given")
    check_disjoint(splits)
    if on not in frame.columns:
        raise ValueError(f"frame has no {on!r} column to assign splits by")
    parts = [
        frame.filter(s.range.predicate(on)).with_columns(
            pl.lit(s.role.value).alias("split_role"), pl.lit(s.fold, dtype=pl.Int32).alias("fold")
        )
        for s in splits
    ]
    return pl.concat(parts)
