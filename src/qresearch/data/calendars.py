"""Trading calendars: when a venue's regular session is open, in UTC.

Two calendars for the MVP (DEVELOPMENT_PLAN.md sec. 3): a 24/7 calendar for crypto and
one US equity calendar. Calendars are *versioned* by id (``XNYS:1``) because a calendar
is data: exchanges add closures (national days of mourning), change early-close rules,
and add holidays (Juneteenth, 2022). A dataset manifest records the calendar id it was
validated against; a rule change is a new id, not an edit.

Session times are computed from local wall-clock rules via ``zoneinfo`` and converted to
UTC per date, which is the only way to get DST right: the NYSE opens at 13:30 UTC in
summer and 14:30 UTC in winter, and a bar timestamped 14:00 UTC is pre-market in one and
mid-session in the other.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterator
from functools import lru_cache
from typing import Final, Protocol
from zoneinfo import ZoneInfo

import polars as pl

from qresearch.config import FrozenModel
from qresearch.time import UTC, UtcDatetime, ensure_utc


class Session(FrozenModel):
    """One regular trading session, ``[open, close)`` in UTC."""

    date: _dt.date
    """The venue-local session date."""

    open: UtcDatetime
    close: UtcDatetime
    early_close: bool = False

    def contains(self, moment: _dt.datetime) -> bool:
        return self.open <= moment < self.close

    @property
    def duration(self) -> _dt.timedelta:
        return self.close - self.open


class TradingCalendar(Protocol):
    @property
    def calendar_id(self) -> str: ...

    def sessions(self, start: _dt.date, end: _dt.date) -> list[Session]:
        """Sessions whose local date lies in ``[start, end]`` (inclusive, dates)."""
        ...

    def session_at(self, moment: _dt.datetime) -> Session | None:
        """The session containing ``moment``, or None if the venue is closed then."""
        ...


# -- 24/7 --------------------------------------------------------------------------------


class AlwaysOpenCalendar:
    """Continuous trading. Sessions are UTC days, purely as a bookkeeping unit."""

    calendar_id: Final = "24x7:1"

    def sessions(self, start: _dt.date, end: _dt.date) -> list[Session]:
        return [self._session(day) for day in _dates(start, end)]

    def session_at(self, moment: _dt.datetime) -> Session:
        return self._session(ensure_utc(moment).date())

    @staticmethod
    def _session(day: _dt.date) -> Session:
        open_ = _dt.datetime.combine(day, _dt.time(0), tzinfo=UTC)
        return Session(date=day, open=open_, close=open_ + _dt.timedelta(days=1))


# -- XNYS --------------------------------------------------------------------------------

_NY: Final = ZoneInfo("America/New_York")
_REGULAR_OPEN: Final = _dt.time(9, 30)
_REGULAR_CLOSE: Final = _dt.time(16, 0)
_EARLY_CLOSE: Final = _dt.time(13, 0)

XNYS_SPECIAL_CLOSURES: Final[frozenset[_dt.date]] = frozenset(
    {
        _dt.date(2001, 9, 11),
        _dt.date(2001, 9, 12),
        _dt.date(2001, 9, 13),
        _dt.date(2001, 9, 14),
        _dt.date(2004, 6, 11),  # Reagan
        _dt.date(2007, 1, 2),  # Ford
        _dt.date(2012, 10, 29),  # Sandy
        _dt.date(2012, 10, 30),
        _dt.date(2018, 12, 5),  # G. H. W. Bush
        _dt.date(2025, 1, 9),  # Carter
    }
)
"""Ad-hoc full closures. Appending one is a calendar version bump."""


def easter(year: int) -> _dt.date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    length = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * length) // 451
    month, day = divmod(h + length - 7 * m + 114, 31)
    return _dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """``n``-th (1-based) ``weekday`` (Mon=0) of the month; ``n=-1`` for the last."""
    if n > 0:
        first = _dt.date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + _dt.timedelta(days=offset + 7 * (n - 1))
    last = _dt.date(year + (month == 12), month % 12 + 1, 1) - _dt.timedelta(days=1)
    return last - _dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: _dt.date, *, saturday_to_friday: bool = True) -> _dt.date | None:
    """NYSE observance: Sunday -> Monday; Saturday -> Friday, except where the exchange
    does not observe (New Year's Day on a Saturday is not moved)."""
    if day.weekday() == 6:
        return day + _dt.timedelta(days=1)
    if day.weekday() == 5:
        return day - _dt.timedelta(days=1) if saturday_to_friday else None
    return day


@lru_cache(maxsize=64)
def xnys_holidays(year: int) -> frozenset[_dt.date]:
    """Full-day NYSE closures in ``year`` under version ``XNYS:1`` rules."""
    days: set[_dt.date | None] = {
        _observed(_dt.date(year, 1, 1), saturday_to_friday=False),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        easter(year) - _dt.timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(_dt.date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(_dt.date(year, 12, 25)),  # Christmas
    }
    if year >= 2022:
        days.add(_observed(_dt.date(year, 6, 19)))  # Juneteenth
    # A New Year's Day observed on Monday for the *next* year can fall in this year? No:
    # Jan 1 on Sunday is observed Jan 2. But Dec 31 is never a holiday. Nothing to add.
    return frozenset(d for d in days if d is not None) | frozenset(
        d for d in XNYS_SPECIAL_CLOSURES if d.year == year
    )


def xnys_early_close(day: _dt.date) -> bool:
    """13:00 closes: day after Thanksgiving; Dec 24 and Jul 3 when they are weekdays
    ahead of a weekday holiday."""
    if day == _nth_weekday(day.year, 11, 3, 4) + _dt.timedelta(days=1):
        return True
    if day.month == 12 and day.day == 24 and day.weekday() < 5:
        return True
    return day.month == 7 and day.day == 3 and day.weekday() < 4  # Mon..Thu


class XNYSCalendar:
    """New York Stock Exchange regular session, rules version 1.

    Regular hours 09:30-16:00 America/New_York; early closes at 13:00. Holiday rules are
    encoded, so any year is computable, but they have only been checked against the
    published calendars for 2020-2026. Use outside that range with care.
    """

    calendar_id: Final = "XNYS:1"

    def is_trading_day(self, day: _dt.date) -> bool:
        return day.weekday() < 5 and day not in xnys_holidays(day.year)

    def sessions(self, start: _dt.date, end: _dt.date) -> list[Session]:
        return [self._session(d) for d in _dates(start, end) if self.is_trading_day(d)]

    def session_at(self, moment: _dt.datetime) -> Session | None:
        local = ensure_utc(moment).astimezone(_NY)
        day = local.date()
        if not self.is_trading_day(day):
            return None
        session = self._session(day)
        return session if session.contains(ensure_utc(moment)) else None

    def _session(self, day: _dt.date) -> Session:
        early = xnys_early_close(day)
        close_time = _EARLY_CLOSE if early else _REGULAR_CLOSE
        return Session(
            date=day,
            open=_dt.datetime.combine(day, _REGULAR_OPEN, tzinfo=_NY).astimezone(UTC),
            close=_dt.datetime.combine(day, close_time, tzinfo=_NY).astimezone(UTC),
            early_close=early,
        )


# -- registry ------------------------------------------------------------------------------

_CALENDARS: Final[dict[str, TradingCalendar]] = {
    AlwaysOpenCalendar.calendar_id: AlwaysOpenCalendar(),
    XNYSCalendar.calendar_id: XNYSCalendar(),
}


def get_calendar(calendar_id: str) -> TradingCalendar:
    try:
        return _CALENDARS[calendar_id]
    except KeyError:
        raise KeyError(f"unknown calendar {calendar_id!r}; known: {sorted(_CALENDARS)}") from None


# -- frame helpers ---------------------------------------------------------------------------

SESSION_COLUMNS: Final = ("session_date", "session_open", "session_close")


def attach_sessions(bars: pl.DataFrame, calendar: TradingCalendar) -> pl.DataFrame:
    """Add ``session_date``, ``session_open``, ``session_close`` to each bar.

    A bar belongs to the session containing its ``bar_start``. Bars outside any session
    (pre/post-market, weekends, holidays) get nulls; session features then null out, and
    the count of such bars is a validation signal for equity datasets.

    Implemented as a backward as-of join of ``bar_start`` against session opens, then
    nulled where the bar starts at or after that session's close.
    """
    if bars.is_empty():
        return bars.with_columns(
            pl.lit(None, dtype=pl.Date).alias("session_date"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("session_open"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("session_close"),
        )
    first = ensure_utc(bars.get_column("bar_start").min())  # type: ignore[arg-type]
    last = ensure_utc(bars.get_column("bar_start").max())  # type: ignore[arg-type]
    # One day of slack either side covers venue-local dates straddling UTC midnight.
    sessions = calendar.sessions(
        first.date() - _dt.timedelta(days=1), last.date() + _dt.timedelta(days=1)
    )
    table = pl.DataFrame(
        {
            "session_date": [s.date for s in sessions],
            "session_open": [s.open for s in sessions],
            "session_close": [s.close for s in sessions],
        },
        schema={
            "session_date": pl.Date,
            "session_open": pl.Datetime("us", "UTC"),
            "session_close": pl.Datetime("us", "UTC"),
        },
    ).sort("session_open")
    original_order = bars.with_row_index("__row")
    joined = (
        original_order.sort("bar_start")
        .join_asof(table, left_on="bar_start", right_on="session_open", strategy="backward")
        .with_columns(
            pl.when(pl.col("bar_start") < pl.col("session_close"))
            .then(pl.col("session_date"))
            .otherwise(None)
            .alias("session_date"),
            pl.when(pl.col("bar_start") < pl.col("session_close"))
            .then(pl.col("session_open"))
            .otherwise(None)
            .alias("session_open"),
            pl.when(pl.col("bar_start") < pl.col("session_close"))
            .then(pl.col("session_close"))
            .otherwise(None)
            .alias("session_close"),
        )
        .sort("__row")
        .drop("__row")
    )
    return joined


def _dates(start: _dt.date, end: _dt.date) -> Iterator[_dt.date]:
    day = start
    while day <= end:
        yield day
        day += _dt.timedelta(days=1)
