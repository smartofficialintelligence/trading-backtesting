"""Session-relative features over an equity session, plus leakage checks."""

from __future__ import annotations

import polars as pl
import pytest
from tests.unit.test_calendars import equity_bars, utc

from qresearch.data.calendars import XNYSCalendar, attach_sessions
from qresearch.features.leakage import assert_future_insensitive, assert_prefix_invariant
from qresearch.features.pipeline import compute_features
from qresearch.features.session import (
    IsEarlyClose,
    MinutesSinceOpen,
    MinutesToClose,
    SessionFraction,
)
from qresearch.features.technical import LaggedReturn

ALL = [MinutesSinceOpen(), MinutesToClose(), SessionFraction(), IsEarlyClose()]


@pytest.fixture
def session_day() -> pl.DataFrame:
    """13:00 .. 20:29 UTC on 2024-03-11: pre-market, full session, post-market."""
    return attach_sessions(equity_bars(450, start=utc(2024, 3, 11, 13, 0)), XNYSCalendar())


def test_values_at_the_open_and_close(session_day: pl.DataFrame) -> None:
    out = compute_features(session_day, ALL)
    first = out.filter(pl.col("bar_start") == utc(2024, 3, 11, 13, 30)).row(0, named=True)
    last = out.filter(pl.col("bar_start") == utc(2024, 3, 11, 19, 59)).row(0, named=True)
    assert first["minutes_since_open"] == 0
    assert first["minutes_to_close"] == 389
    assert first["session_fraction"] == pytest.approx(0.0)
    assert last["minutes_since_open"] == 389
    assert last["minutes_to_close"] == 0
    assert last["session_fraction"] == pytest.approx(389 / 390)
    assert first["is_early_close"] == 0


def test_outside_the_session_everything_is_null(session_day: pl.DataFrame) -> None:
    out = compute_features(session_day, ALL)
    outside = out.filter(
        (pl.col("bar_start") < utc(2024, 3, 11, 13, 30))
        | (pl.col("bar_start") >= utc(2024, 3, 11, 20, 0))
    )
    assert outside.height == 60
    for column in ("minutes_since_open", "minutes_to_close", "session_fraction", "is_early_close"):
        assert outside.get_column(column).null_count() == 60


def test_early_close_day() -> None:
    bars = attach_sessions(equity_bars(240, start=utc(2024, 11, 29, 14, 30)), XNYSCalendar())
    out = compute_features(bars, ALL)
    inside = out.filter(pl.col("is_early_close").is_not_null())
    assert inside.height == 210, "09:30 .. 13:00 EST"
    assert set(inside.get_column("is_early_close").to_list()) == {1}
    assert inside.get_column("minutes_to_close").min() == 0
    assert inside.get_column("session_fraction").max() == pytest.approx(209 / 210)


def test_forgetting_to_attach_sessions_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="require bar columns \\['session_open'\\]"):
        compute_features(equity_bars(5, start=utc(2024, 3, 11, 13, 30)), [MinutesSinceOpen()])


def test_session_features_are_available_with_their_bar(session_day: pl.DataFrame) -> None:
    out = compute_features(session_day, ALL)
    assert (
        out.get_column("available_at").to_list() == session_day.get_column("available_at").to_list()
    )


def test_session_features_pass_the_leakage_checks() -> None:
    bars = attach_sessions(
        equity_bars(450, start=utc(2024, 3, 11, 13, 0)).with_columns(
            pl.when(pl.col("bar_start") == utc(2024, 3, 11, 15, 0))
            .then(pl.col("available_at") + pl.duration(minutes=5))
            .otherwise(pl.col("available_at"))
            .alias("available_at")
        ),
        XNYSCalendar(),
    )

    def compute(b: pl.DataFrame) -> pl.DataFrame:
        return compute_features(b, [*ALL, LaggedReturn(1)])

    assert_prefix_invariant(compute, bars)
    assert_future_insensitive(compute, bars)
