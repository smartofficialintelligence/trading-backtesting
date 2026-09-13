"""Walk-forward fold generation with warm-up, purge, and embargo.

Folds are chronological: every training range ends before the validation and test
ranges of the same fold begin. Between them sits a **purge** gap at least as long as the
label horizon, so no training label can overlap a test observation (ARCHITECTURE.md
sec. 9, "label overlap"). After a fold's test range an **embargo** gap is recorded; rows
inside it are excluded from the training set of every later fold that would otherwise
cover them, so a later fit cannot learn from observations adjacent to an earlier test.

Ranges are periods of *knowledge*: rows are assigned by ``available_at`` (see
:mod:`qresearch.research.splits`).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from enum import StrEnum
from typing import Self

import polars as pl
from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.research.splits import DataSplit, SplitRole, TimeRange

_ZERO = _dt.timedelta(0)


class WalkForwardKind(StrEnum):
    ROLLING = "rolling"
    """Fixed-length training window that slides forward."""

    EXPANDING = "expanding"
    """Training window anchored at the start that grows each fold."""


class WalkForwardPlan(FrozenModel):
    kind: WalkForwardKind = WalkForwardKind.ROLLING
    train: _dt.timedelta = Field(gt=_ZERO)
    validation: _dt.timedelta = Field(default=_ZERO, ge=_ZERO)
    test: _dt.timedelta = Field(gt=_ZERO)
    step: _dt.timedelta | None = None
    """How far the fold advances. Defaults to ``validation + test`` (no overlap of
    evaluation periods between folds)."""

    warmup: _dt.timedelta = Field(default=_ZERO, ge=_ZERO)
    """Data before the training range supplied for lookbacks only."""

    purge: _dt.timedelta = Field(default=_ZERO, ge=_ZERO)
    label_horizon: _dt.timedelta | None = None
    """Longest horizon any label or lookforward uses. The effective purge is
    ``max(purge, label_horizon)``. Required unless ``purge`` is set explicitly, so the
    gap is always a stated decision rather than an omission."""

    embargo: _dt.timedelta = Field(default=_ZERO, ge=_ZERO)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.step is not None and self.step <= _ZERO:
            raise ValueError("step must be positive")
        if self.label_horizon is not None and self.label_horizon < _ZERO:
            raise ValueError("label_horizon must be non-negative")
        if self.purge == _ZERO and self.label_horizon is None:
            raise ValueError(
                "state the purge gap: set label_horizon (purge is derived from it) or "
                "purge explicitly; a zero gap must be a decision, not a default"
            )
        return self

    @property
    def effective_purge(self) -> _dt.timedelta:
        return max(self.purge, self.label_horizon or _ZERO)

    @property
    def effective_step(self) -> _dt.timedelta:
        return self.step if self.step is not None else self.validation + self.test


class Fold(FrozenModel):
    index: int = Field(ge=0)
    warmup: TimeRange | None
    train: TimeRange
    purge_after_train: TimeRange | None
    validation: TimeRange | None
    purge_after_validation: TimeRange | None
    test: TimeRange
    embargo: TimeRange | None
    train_exclusions: tuple[TimeRange, ...] = ()
    """Embargo ranges of earlier folds that fall inside this fold's training range."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        cursor = self.train.end
        for name, part in (
            ("purge_after_train", self.purge_after_train),
            ("validation", self.validation),
            ("purge_after_validation", self.purge_after_validation),
            ("test", self.test),
        ):
            if part is None:
                continue
            if part.start < cursor:
                raise ValueError(
                    f"fold {self.index}: {name} starts before the preceding range ends"
                )
            cursor = part.end
        if self.warmup is not None and self.warmup.end > self.train.start:
            raise ValueError(f"fold {self.index}: warmup overlaps training")
        return self

    @property
    def span(self) -> TimeRange:
        """From the earliest data needed to the end of the test range."""
        start = self.warmup.start if self.warmup is not None else self.train.start
        return TimeRange(start=start, end=self.test.end)

    def splits(self) -> tuple[DataSplit, ...]:
        parts: list[tuple[SplitRole, TimeRange | None]] = [
            (SplitRole.WARMUP, self.warmup),
            (SplitRole.TRAIN, self.train),
            (SplitRole.PURGE, self.purge_after_train),
            (SplitRole.VALIDATION, self.validation),
            (SplitRole.PURGE, self.purge_after_validation),
            (SplitRole.TEST, self.test),
            (SplitRole.EMBARGO, self.embargo),
        ]
        return tuple(
            DataSplit(role=r, range=rng, fold=self.index) for r, rng in parts if rng is not None
        )

    def training_predicate(self, column: str = "available_at") -> pl.Expr:
        """Rows usable for fitting: inside train, outside every exclusion."""
        expr = self.train.predicate(column)
        for excluded in self.train_exclusions:
            expr = expr & ~excluded.predicate(column)
        return expr


def generate_folds(
    plan: WalkForwardPlan, span: TimeRange, exclude: Sequence[TimeRange] = ()
) -> list[Fold]:
    """All folds whose test range ends inside ``span``, minus any that touch ``exclude``.

    A fold touches an excluded range if any part of its span -- warm-up through test --
    overlaps it. Such folds are skipped, not shortened, and the rest are numbered from 0.
    Raises if no fold remains.
    """
    purge = plan.effective_purge
    first_train_start = span.start + plan.warmup
    folds: list[Fold] = []
    embargoes: list[TimeRange] = []
    skipped = 0
    k = 0
    while True:
        train_end = first_train_start + plan.train + k * plan.effective_step
        train_start = (
            train_end - plan.train if plan.kind is WalkForwardKind.ROLLING else first_train_start
        )
        cursor = train_end
        purge1 = _range(cursor, purge)
        cursor = purge1.end if purge1 else cursor
        validation = _range(cursor, plan.validation)
        cursor = validation.end if validation else cursor
        purge2 = _range(cursor, purge) if validation is not None else None
        cursor = purge2.end if purge2 else cursor
        test = TimeRange(start=cursor, end=cursor + plan.test)
        if test.end > span.end:
            break
        warmup = (
            TimeRange(start=train_start - plan.warmup, end=train_start)
            if plan.warmup > _ZERO
            else None
        )
        train = TimeRange(start=train_start, end=train_end)
        reach = TimeRange(start=warmup.start if warmup else train_start, end=test.end)
        if any(reach.overlaps(excluded) for excluded in exclude):
            skipped += 1
            k += 1
            continue
        fold = Fold(
            index=len(folds),
            warmup=warmup,
            train=train,
            purge_after_train=purge1,
            validation=validation,
            purge_after_validation=purge2,
            test=test,
            embargo=_range(test.end, plan.embargo),
            train_exclusions=tuple(e for e in embargoes if e.overlaps(train)),
        )
        folds.append(fold)
        if fold.embargo is not None:
            embargoes.append(fold.embargo)
        k += 1
    if not folds and skipped:
        raise ValueError(f"all {skipped} fold(s) touch an excluded range; none remain")
    if not folds:
        needed = plan.warmup + plan.train + purge + plan.validation + plan.test
        if plan.validation > _ZERO:
            needed += purge
        raise ValueError(
            f"the plan needs at least {needed} of data but the span is {span.duration}; "
            "no fold fits"
        )
    return folds


class FixedSplitPlan(FrozenModel):
    """One explicit train / validation / test fold. Gaps must respect the purge."""

    train: TimeRange
    validation: TimeRange | None = None
    test: TimeRange
    warmup: _dt.timedelta = Field(default=_ZERO, ge=_ZERO)
    purge: _dt.timedelta = Field(default=_ZERO, ge=_ZERO)
    label_horizon: _dt.timedelta | None = None

    @property
    def effective_purge(self) -> _dt.timedelta:
        return max(self.purge, self.label_horizon or _ZERO)

    def fold(self) -> Fold:
        purge = self.effective_purge
        after_train = self.validation.start if self.validation is not None else self.test.start
        if after_train < self.train.end:
            raise ValueError(
                f"{'validation' if self.validation else 'test'} starts before training ends"
            )
        if after_train - self.train.end < purge:
            raise ValueError(
                f"gap after training ({after_train - self.train.end}) is shorter than the "
                f"purge ({purge})"
            )
        if self.validation is not None and self.test.start - self.validation.end < purge:
            raise ValueError("gap after validation is shorter than the purge")
        return Fold(
            index=0,
            warmup=TimeRange(start=self.train.start - self.warmup, end=self.train.start)
            if self.warmup > _ZERO
            else None,
            train=self.train,
            purge_after_train=TimeRange(start=self.train.end, end=after_train)
            if after_train > self.train.end
            else None,
            validation=self.validation,
            purge_after_validation=(
                TimeRange(start=self.validation.end, end=self.test.start)
                if self.validation is not None and self.test.start > self.validation.end
                else None
            ),
            test=self.test,
            embargo=None,
        )


def _range(start: _dt.datetime, length: _dt.timedelta) -> TimeRange | None:
    return TimeRange(start=start, end=start + length) if length > _ZERO else None
