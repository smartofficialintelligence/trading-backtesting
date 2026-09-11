"""Fold generation: chronology, purge, embargo, warm-up, and the no-overlap guarantee."""

from __future__ import annotations

import datetime as dt
from itertools import pairwise

import polars as pl
import pytest
from pydantic import ValidationError
from tests.unit.test_resample import minute_bars

from qresearch.research.splits import SplitRole, TimeRange, check_disjoint
from qresearch.research.walk_forward import (
    FixedSplitPlan,
    WalkForwardKind,
    WalkForwardPlan,
    generate_folds,
)

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)
H = dt.timedelta(hours=1)
M = dt.timedelta(minutes=1)


def span(hours: float) -> TimeRange:
    return TimeRange(start=T0, end=T0 + hours * H)


def test_the_purge_must_be_a_stated_decision() -> None:
    with pytest.raises(ValidationError, match="state the purge gap"):
        WalkForwardPlan(train=2 * H, test=H)
    WalkForwardPlan(train=2 * H, test=H, purge=M)
    WalkForwardPlan(train=2 * H, test=H, label_horizon=dt.timedelta(0))


def test_effective_purge_is_the_larger_of_purge_and_label_horizon() -> None:
    assert (
        WalkForwardPlan(train=H, test=H, purge=5 * M, label_horizon=10 * M).effective_purge
        == 10 * M
    )
    assert (
        WalkForwardPlan(train=H, test=H, purge=15 * M, label_horizon=10 * M).effective_purge
        == 15 * M
    )


def test_rolling_folds_slide_by_the_evaluation_length() -> None:
    plan = WalkForwardPlan(train=2 * H, test=H, purge=10 * M)
    folds = generate_folds(plan, span(6))
    assert [f.index for f in folds] == [0, 1, 2]
    f0, f1 = folds[0], folds[1]
    assert f0.train == TimeRange(start=T0, end=T0 + 2 * H)
    assert f0.purge_after_train == TimeRange(start=T0 + 2 * H, end=T0 + 2 * H + 10 * M)
    assert f0.test == TimeRange(start=T0 + 2 * H + 10 * M, end=T0 + 3 * H + 10 * M)
    assert f1.train == TimeRange(start=T0 + H, end=T0 + 3 * H)
    assert f1.test.start == f0.test.start + H, "step defaults to the test length"
    assert all(f.train.duration == 2 * H for f in folds)


def test_expanding_folds_keep_the_anchor() -> None:
    plan = WalkForwardPlan(
        kind=WalkForwardKind.EXPANDING, train=2 * H, test=H, purge=0 * M, label_horizon=0 * M
    )
    folds = generate_folds(plan, span(6))
    assert [f.train.start for f in folds] == [T0] * len(folds)
    assert [f.train.duration for f in folds] == [2 * H, 3 * H, 4 * H, 5 * H]


def test_validation_gets_its_own_purge() -> None:
    plan = WalkForwardPlan(train=2 * H, validation=H, test=H, purge=10 * M)
    fold = generate_folds(plan, span(8))[0]
    assert fold.validation == TimeRange(start=T0 + 2 * H + 10 * M, end=T0 + 3 * H + 10 * M)
    assert fold.purge_after_validation == TimeRange(
        start=fold.validation.end, end=fold.validation.end + 10 * M
    )
    assert fold.test.start == fold.validation.end + 10 * M
    assert generate_folds(plan, span(8))[1].test.start == fold.test.start + 2 * H, (
        "step = validation + test"
    )


def test_warmup_precedes_training_and_is_inside_the_span() -> None:
    plan = WalkForwardPlan(train=2 * H, test=H, purge=M, warmup=30 * M)
    fold = generate_folds(plan, span(6))[0]
    assert fold.warmup == TimeRange(start=T0, end=T0 + 30 * M)
    assert fold.train.start == T0 + 30 * M
    assert fold.span.start == T0


def test_embargo_after_a_test_is_excluded_from_later_training() -> None:
    plan = WalkForwardPlan(train=3 * H, test=H, purge=M, embargo=20 * M)
    folds = generate_folds(plan, span(8))
    f0, f1, f2 = folds[0], folds[1], folds[2]
    assert f0.embargo == TimeRange(start=f0.test.end, end=f0.test.end + 20 * M)
    assert f0.train_exclusions == ()
    # Fold 1 trains on [1h, 4h): it ends before the embargo begins at 4h01, so no exclusion.
    assert not f1.train.overlaps(f0.embargo) and f1.train_exclusions == ()
    # Fold 2 trains on [2h, 5h), which covers fold 0's embargo.
    assert f2.train.overlaps(f0.embargo)
    assert f2.train_exclusions == (f0.embargo,)


def test_training_predicate_honours_exclusions() -> None:
    plan = WalkForwardPlan(train=3 * H, test=H, purge=M, embargo=20 * M)
    folds = generate_folds(plan, span(8))
    f2 = folds[2]
    bars = minute_bars(8 * 60)
    selected = bars.filter(f2.training_predicate())
    in_embargo = selected.filter(folds[0].embargo.predicate())  # type: ignore[union-attr]
    assert in_embargo.is_empty()
    assert selected.height > 0
    naive = bars.filter(f2.train.predicate())
    assert naive.height - selected.height == 20, "the 20 embargoed minutes"


def test_splits_within_a_fold_are_disjoint_and_ordered() -> None:
    plan = WalkForwardPlan(
        train=2 * H, validation=H, test=H, purge=10 * M, warmup=15 * M, embargo=5 * M
    )
    for fold in generate_folds(plan, span(10)):
        splits = fold.splits()
        check_disjoint(splits)
        roles = [s.role for s in splits]
        assert roles == [
            SplitRole.WARMUP,
            SplitRole.TRAIN,
            SplitRole.PURGE,
            SplitRole.VALIDATION,
            SplitRole.PURGE,
            SplitRole.TEST,
            SplitRole.EMBARGO,
        ]
        assert all(a.range.end <= b.range.start for a, b in pairwise(splits))


def test_no_row_can_be_in_train_and_test_of_the_same_fold() -> None:
    """The point of the whole module, checked on actual rows with a label horizon: no
    training row's label window reaches into the test range."""
    horizon = 15 * M
    plan = WalkForwardPlan(train=2 * H, test=H, label_horizon=horizon)
    bars = minute_bars(6 * 60)
    for fold in generate_folds(plan, span(6)):
        train_rows = bars.filter(fold.training_predicate())
        last_label_end = train_rows.get_column("available_at").max() + horizon
        assert last_label_end <= fold.test.start


def test_a_span_too_short_for_one_fold_is_an_error() -> None:
    with pytest.raises(ValueError, match="no fold fits"):
        generate_folds(WalkForwardPlan(train=2 * H, test=H, purge=M), span(2.5))


def test_fixed_plan_checks_the_purge_gap() -> None:
    plan = FixedSplitPlan(
        train=TimeRange(start=T0, end=T0 + 2 * H),
        test=TimeRange(start=T0 + 2 * H + 5 * M, end=T0 + 3 * H),
        label_horizon=10 * M,
    )
    with pytest.raises(ValueError, match="shorter than the purge"):
        plan.fold()
    ok = plan.model_copy(update={"label_horizon": 5 * M}).fold()
    assert ok.purge_after_train == TimeRange(start=T0 + 2 * H, end=T0 + 2 * H + 5 * M)
    assert [s.role for s in ok.splits()] == [SplitRole.TRAIN, SplitRole.PURGE, SplitRole.TEST]


def test_fold_rejects_out_of_order_ranges() -> None:
    with pytest.raises(ValueError, match="starts before"):
        FixedSplitPlan(
            train=TimeRange(start=T0, end=T0 + 2 * H),
            test=TimeRange(start=T0 + H, end=T0 + 3 * H),
            purge=dt.timedelta(0),
        ).fold()


def test_folds_are_serialisable() -> None:
    fold = generate_folds(WalkForwardPlan(train=2 * H, test=H, purge=M), span(4))[0]
    assert type(fold).model_validate_json(fold.model_dump_json()) == fold
    assert isinstance(fold.training_predicate(), pl.Expr)
