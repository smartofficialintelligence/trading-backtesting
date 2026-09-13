"""Sealed test windows: the registry, its guarded ranges, and the committed windows."""

from __future__ import annotations

import datetime as dt
from itertools import pairwise
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from qresearch.research.holdout import (
    DEFAULT_REGISTRY,
    REGISTRY_ENV,
    SealedWindow,
    SealedWindows,
)
from qresearch.research.splits import TimeRange

T0 = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
D = dt.timedelta(days=1)


def window(name: str, start_day: int, end_day: int, gap_days: int = 2) -> SealedWindow:
    return SealedWindow(
        name=name,
        range=TimeRange(start=T0 + start_day * D, end=T0 + end_day * D),
        gap=gap_days * D,
    )


def test_the_guarded_range_adds_the_gap_on_both_sides() -> None:
    assert window("w", 10, 20, gap_days=3).guarded == TimeRange(start=T0 + 7 * D, end=T0 + 23 * D)


def test_a_span_in_the_gap_touches_the_window_and_one_past_it_does_not() -> None:
    sealed = SealedWindows(windows=(window("w", 10, 20, gap_days=3),))
    assert sealed.touching(TimeRange(start=T0 + 21 * D, end=T0 + 22 * D)), "inside the gap"
    assert not sealed.touching(TimeRange(start=T0 + 23 * D, end=T0 + 30 * D)), "half-open end"
    assert not sealed.touching(TimeRange(start=T0, end=T0 + 7 * D)), "half-open start"


def test_names_are_unique_and_lookups_must_name_a_real_window() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SealedWindows(windows=(window("w", 1, 2), window("w", 5, 6)))
    with pytest.raises(ValueError, match="no sealed window named 'x'"):
        SealedWindows(windows=(window("w", 1, 2),)).get("x")


def test_excluded_ranges_leave_out_the_windows_being_kept() -> None:
    sealed = SealedWindows(windows=(window("a", 1, 2), window("b", 5, 6)))
    assert sealed.excluded_ranges() == (sealed.get("a").guarded, sealed.get("b").guarded)
    assert sealed.excluded_ranges(keep=("a",)) == (sealed.get("b").guarded,)


def test_the_development_predicate_drops_rows_inside_windows_and_their_gaps() -> None:
    sealed = SealedWindows(windows=(window("w", 10, 20, gap_days=3),))
    frame = pl.DataFrame({"bar_start": [T0 + d * D for d in (0, 7, 15, 22, 23, 40)]})
    kept = frame.filter(sealed.development_predicate())["bar_start"].to_list()
    assert kept == [T0, T0 + 23 * D, T0 + 40 * D]


def test_the_default_registry_honours_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sealed.yaml"
    path.write_text(
        "windows:\n"
        "- name: w\n"
        "  range: {start: '2025-01-11T00:00:00Z', end: '2025-01-21T00:00:00Z'}\n"
        "  gap: P3D\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(REGISTRY_ENV, str(path))
    assert SealedWindows.default().get("w").guarded == TimeRange(start=T0 + 7 * D, end=T0 + 23 * D)
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "missing.yaml"))
    assert SealedWindows.default() == SealedWindows()


def test_the_committed_windows_are_the_ones_recorded_in_d70() -> None:
    """A tripwire: a window must not move once anything has been evaluated in it."""
    sealed = SealedWindows.load(DEFAULT_REGISTRY)
    assert {w.name: (w.range.start.date(), w.range.end.date(), w.gap) for w in sealed.windows} == {
        "bull-2025": (dt.date(2025, 5, 5), dt.date(2025, 8, 4), 7 * D),
        "bear-2026": (dt.date(2026, 1, 19), dt.date(2026, 4, 20), 7 * D),
        "flat-2026": (dt.date(2026, 6, 19), dt.date(2026, 8, 19), 7 * D),
    }
    ranges = sorted((w.guarded for w in sealed.windows), key=lambda r: r.start)
    assert not any(a.overlaps(b) for a, b in pairwise(ranges))
