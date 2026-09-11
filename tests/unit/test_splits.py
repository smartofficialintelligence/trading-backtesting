"""Time ranges and role assignment."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from pydantic import ValidationError
from tests.unit.test_resample import minute_bars

from qresearch.research.splits import (
    DataSplit,
    SplitRole,
    TimeRange,
    assign_roles,
    check_disjoint,
    select_split,
)

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)


def at(minute: int, second: int = 0) -> dt.datetime:
    return T0 + dt.timedelta(minutes=minute, seconds=second)


def rng(a: int, b: int) -> TimeRange:
    return TimeRange(start=at(a), end=at(b))


def test_ranges_are_half_open() -> None:
    r = rng(0, 10)
    assert r.contains(at(0)) and r.contains(at(9, 59))
    assert not r.contains(at(10))
    assert r.duration == dt.timedelta(minutes=10)


def test_empty_or_inverted_ranges_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must precede"):
        rng(5, 5)
    with pytest.raises(ValidationError, match="must precede"):
        rng(6, 5)


def test_adjacent_ranges_do_not_overlap() -> None:
    assert not rng(0, 10).overlaps(rng(10, 20))
    assert rng(0, 10).overlaps(rng(9, 20))


def test_only_train_is_fittable() -> None:
    for role in SplitRole:
        assert DataSplit(role=role, range=rng(0, 1)).fittable is (role is SplitRole.TRAIN)


def test_overlapping_splits_in_one_fold_are_refused() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        check_disjoint(
            [
                DataSplit(role=SplitRole.TRAIN, range=rng(0, 10)),
                DataSplit(role=SplitRole.TEST, range=rng(9, 20)),
            ]
        )


def test_overlap_across_folds_is_fine() -> None:
    check_disjoint(
        [
            DataSplit(role=SplitRole.TRAIN, range=rng(0, 10), fold=0),
            DataSplit(role=SplitRole.TRAIN, range=rng(5, 15), fold=1),
        ]
    )


def test_rows_are_assigned_by_availability_not_by_bar_start() -> None:
    """Bar 9 publishes 9 minutes late (available 00:19). A split ending at 00:15 covers
    its interval but not its publication; it is not knowable in that split."""
    bars = minute_bars(20, late={9: dt.timedelta(minutes=9)})
    train = DataSplit(role=SplitRole.TRAIN, range=rng(0, 15))
    selected = select_split(bars, train)
    starts = set(selected.get_column("bar_start").to_list())
    assert at(9) not in starts
    assert at(13) in starts, "available 00:14:02 < 00:15"
    assert at(14) not in starts, "available 00:15:02 >= 00:15"


def test_assign_roles_tags_and_drops_unassigned() -> None:
    bars = minute_bars(30)
    tagged = assign_roles(
        bars,
        [
            DataSplit(role=SplitRole.TRAIN, range=rng(0, 10)),
            DataSplit(role=SplitRole.PURGE, range=rng(10, 12)),
            DataSplit(role=SplitRole.TEST, range=rng(12, 20)),
        ],
    )
    counts = dict(tagged.group_by("split_role").len().rows())
    assert counts == {"train": 9, "purge": 2, "test": 8}, "availability is bar_end + 2s"
    assert tagged.get_column("fold").dtype == pl.Int32
    assert tagged.height < bars.height, "rows after 00:20 are dropped"


def test_assign_roles_needs_the_timestamp_column() -> None:
    with pytest.raises(ValueError, match="no 'available_at'"):
        assign_roles(
            minute_bars(3).drop("available_at"), [DataSplit(role=SplitRole.TRAIN, range=rng(0, 1))]
        )
