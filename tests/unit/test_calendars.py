"""Calendars: holidays, observance, early closes, DST, and session lookup."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.unit.test_resample import minute_bars

from qresearch.data.calendars import (
    AlwaysOpenCalendar,
    XNYSCalendar,
    attach_sessions,
    easter,
    get_calendar,
    xnys_early_close,
    xnys_holidays,
)

D = dt.date
UTC = dt.UTC


def utc(y: int, mo: int, d: int, h: int, mi: int = 0) -> dt.datetime:
    return dt.datetime(y, mo, d, h, mi, tzinfo=UTC)


# -- 24/7 ------------------------------------------------------------------------------


def test_always_open_sessions_are_utc_days() -> None:
    cal = AlwaysOpenCalendar()
    sessions = cal.sessions(D(2024, 3, 4), D(2024, 3, 5))
    assert [s.date for s in sessions] == [D(2024, 3, 4), D(2024, 3, 5)]
    assert sessions[0].open == utc(2024, 3, 4, 0) and sessions[0].close == utc(2024, 3, 5, 0)
    assert cal.session_at(utc(2024, 3, 4, 23, 59)).date == D(2024, 3, 4)


# -- easter ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2023, D(2023, 4, 9)), (2024, D(2024, 3, 31)), (2025, D(2025, 4, 20)), (2026, D(2026, 4, 5))],
)
def test_easter(year: int, expected: dt.date) -> None:
    assert easter(year) == expected


# -- holidays ---------------------------------------------------------------------------


def test_2024_holidays_match_the_published_calendar() -> None:
    assert xnys_holidays(2024) == frozenset(
        {
            D(2024, 1, 1),
            D(2024, 1, 15),
            D(2024, 2, 19),
            D(2024, 3, 29),
            D(2024, 5, 27),
            D(2024, 6, 19),
            D(2024, 7, 4),
            D(2024, 9, 2),
            D(2024, 11, 28),
            D(2024, 12, 25),
        }
    )


def test_2025_holidays_include_the_carter_closure() -> None:
    days = xnys_holidays(2025)
    assert D(2025, 1, 9) in days
    assert D(2025, 4, 18) in days, "Good Friday"
    assert D(2025, 6, 19) in days


def test_observance_rules() -> None:
    assert D(2022, 12, 26) in xnys_holidays(2022), "Christmas on Sunday -> Monday"
    assert D(2021, 7, 5) in xnys_holidays(2021), "July 4 on Sunday -> Monday"
    assert D(2020, 7, 3) in xnys_holidays(2020), "July 4 on Saturday -> Friday"
    assert D(2021, 12, 31) not in xnys_holidays(2021), "Jan 1 2022 on Saturday: not observed"
    assert D(2022, 1, 1) not in xnys_holidays(2022)
    assert D(2023, 1, 2) in xnys_holidays(2023), "Jan 1 2023 on Sunday -> Monday"


def test_juneteenth_starts_in_2022() -> None:
    assert D(2021, 6, 18) not in xnys_holidays(2021)
    assert D(2022, 6, 20) in xnys_holidays(2022), "June 19 2022 on Sunday -> Monday"


# -- early closes ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "early"),
    [
        (D(2024, 11, 29), True),  # day after Thanksgiving
        (D(2024, 12, 24), True),  # Tuesday
        (D(2023, 12, 24), False),  # Sunday: not a session at all
        (D(2024, 7, 3), True),  # Wednesday, July 4 Thursday
        (D(2026, 7, 3), False),  # Friday: it is the observed holiday (July 4 Saturday)
        (D(2024, 3, 11), False),
    ],
)
def test_early_close_rules(day: dt.date, early: bool) -> None:
    assert xnys_early_close(day) is early


def test_early_close_session_ends_at_1300_local() -> None:
    session = XNYSCalendar().sessions(D(2024, 11, 29), D(2024, 11, 29))[0]
    assert session.early_close
    assert session.close == utc(2024, 11, 29, 18, 0), "13:00 EST"
    assert session.duration == dt.timedelta(hours=3, minutes=30)


# -- DST --------------------------------------------------------------------------------


def test_open_shifts_by_an_hour_across_the_spring_transition() -> None:
    cal = XNYSCalendar()
    before = cal.sessions(D(2024, 3, 8), D(2024, 3, 8))[0]  # Friday, EST
    after = cal.sessions(D(2024, 3, 11), D(2024, 3, 11))[0]  # Monday, EDT
    assert before.open == utc(2024, 3, 8, 14, 30)
    assert after.open == utc(2024, 3, 11, 13, 30)
    assert before.duration == after.duration == dt.timedelta(hours=6, minutes=30)


def test_open_shifts_back_across_the_autumn_transition() -> None:
    cal = XNYSCalendar()
    assert cal.sessions(D(2024, 11, 1), D(2024, 11, 1))[0].open == utc(2024, 11, 1, 13, 30)
    assert cal.sessions(D(2024, 11, 4), D(2024, 11, 4))[0].open == utc(2024, 11, 4, 14, 30)


# -- session_at ---------------------------------------------------------------------------


def test_session_lookup_is_half_open_and_calendar_aware() -> None:
    cal = XNYSCalendar()
    assert cal.session_at(utc(2024, 3, 11, 13, 30)) is not None
    assert cal.session_at(utc(2024, 3, 11, 13, 29)) is None, "pre-market"
    assert cal.session_at(utc(2024, 3, 11, 19, 59)) is not None
    assert cal.session_at(utc(2024, 3, 11, 20, 0)) is None, "close is exclusive"
    assert cal.session_at(utc(2024, 3, 9, 15, 0)) is None, "Saturday"
    assert cal.session_at(utc(2024, 3, 29, 15, 0)) is None, "Good Friday"


def test_weekend_sessions_are_absent() -> None:
    sessions = XNYSCalendar().sessions(D(2024, 3, 8), D(2024, 3, 11))
    assert [s.date for s in sessions] == [D(2024, 3, 8), D(2024, 3, 11)]


def test_registry() -> None:
    assert get_calendar("24x7:1").calendar_id == "24x7:1"
    assert get_calendar("XNYS:1").calendar_id == "XNYS:1"
    with pytest.raises(KeyError, match="unknown calendar"):
        get_calendar("XNYS:99")


# -- attach_sessions -------------------------------------------------------------------------


def equity_bars(count: int, *, start: dt.datetime) -> pl.DataFrame:
    """``minute_bars`` shifted so bar 0 starts at ``start``."""
    shift = start - dt.datetime(2024, 3, 4, tzinfo=UTC)
    micros = shift // dt.timedelta(microseconds=1)
    return minute_bars(count).with_columns(
        *[
            (pl.col(c) + pl.duration(microseconds=micros)).alias(c)
            for c in ("bar_start", "bar_end", "available_at")
        ]
    )


def test_attach_sessions_maps_bars_to_their_session() -> None:
    # 13:00 .. 20:29 UTC on 2024-03-11 (EDT): 30 pre-market, 390 session, 30 post.
    bars = equity_bars(450, start=utc(2024, 3, 11, 13, 0))
    out = attach_sessions(bars, XNYSCalendar())
    inside = out.filter(pl.col("session_open").is_not_null())
    assert inside.height == 390
    assert inside.get_column("bar_start").min() == utc(2024, 3, 11, 13, 30)
    assert inside.get_column("bar_start").max() == utc(2024, 3, 11, 19, 59)
    assert set(inside.get_column("session_date").to_list()) == {D(2024, 3, 11)}
    assert out.get_column("session_close").drop_nulls().unique().to_list() == [
        utc(2024, 3, 11, 20, 0)
    ]


def test_attach_sessions_preserves_row_order_and_count() -> None:
    bars = equity_bars(10, start=utc(2024, 3, 11, 13, 25)).sample(
        fraction=1.0, shuffle=True, seed=1
    )
    out = attach_sessions(bars, XNYSCalendar())
    assert out.height == bars.height
    assert out.get_column("bar_start").to_list() == bars.get_column("bar_start").to_list()


def test_attach_sessions_across_a_dst_boundary() -> None:
    """A 14:00 UTC bar is mid-session on Friday (EST) and pre-market on Monday (EDT)."""
    friday = equity_bars(1, start=utc(2024, 3, 8, 14, 0))
    monday = equity_bars(1, start=utc(2024, 3, 11, 14, 0))
    assert attach_sessions(friday, XNYSCalendar()).item(0, "session_open") is None
    assert attach_sessions(monday, XNYSCalendar()).item(0, "session_open") == utc(
        2024, 3, 11, 13, 30
    )


def test_attach_sessions_on_the_always_open_calendar_never_nulls() -> None:
    out = attach_sessions(minute_bars(5), AlwaysOpenCalendar())
    assert out.get_column("session_open").null_count() == 0


def test_attach_sessions_on_an_empty_frame() -> None:
    out = attach_sessions(minute_bars(3).clear(), XNYSCalendar())
    assert out.height == 0 and "session_close" in out.columns
