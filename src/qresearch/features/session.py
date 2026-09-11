"""Session-relative time features.

These read the ``session_open`` / ``session_close`` columns that
:func:`qresearch.data.calendars.attach_sessions` adds to a bar frame, so the calendar
lookup (a data step) stays separate from the feature (an expression). Forgetting to
attach sessions is a clear pipeline error ("features require bar columns
['session_open']"), not a silently wrong feature.

All are zero-lookback and read only the current bar's own timestamps and its session
bounds, which are known before the session starts -- causal by construction. Outside a
session every value is null.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qresearch.features.contracts import FeatureSpec


def _qualname(obj: object) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


_MINUTE_US = 60_000_000


@dataclass(frozen=True, slots=True)
class MinutesSinceOpen:
    """Whole minutes from the session open to ``bar_start``. 0 for the opening bar."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="minutes_since_open",
            implementation=_qualname(self),
            inputs=("bar_start", "session_open"),
            lookback_bars=0,
        )

    def expression(self) -> pl.Expr:
        return (
            (pl.col("bar_start") - pl.col("session_open")).dt.total_microseconds() // _MINUTE_US
        ).cast(pl.Int32)


@dataclass(frozen=True, slots=True)
class MinutesToClose:
    """Whole minutes from ``bar_end`` to the session close. 0 for the closing bar."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="minutes_to_close",
            implementation=_qualname(self),
            inputs=("bar_end", "session_close"),
            lookback_bars=0,
        )

    def expression(self) -> pl.Expr:
        return (
            (pl.col("session_close") - pl.col("bar_end")).dt.total_microseconds() // _MINUTE_US
        ).cast(pl.Int32)


@dataclass(frozen=True, slots=True)
class SessionFraction:
    """Fraction of the session elapsed at ``bar_start``, in ``[0, 1)``.

    Comparable across regular and early-close days, which ``minutes_since_open`` is not.
    """

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="session_fraction",
            implementation=_qualname(self),
            inputs=("bar_start", "session_open", "session_close"),
            lookback_bars=0,
        )

    def expression(self) -> pl.Expr:
        elapsed = (pl.col("bar_start") - pl.col("session_open")).dt.total_microseconds()
        length = (pl.col("session_close") - pl.col("session_open")).dt.total_microseconds()
        return elapsed.cast(pl.Float64) / length.cast(pl.Float64)


@dataclass(frozen=True, slots=True)
class IsEarlyClose:
    """1 when the bar's session closes early, else 0; null outside a session."""

    regular_minutes: int = 390

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="is_early_close",
            implementation=_qualname(self),
            inputs=("session_open", "session_close"),
            lookback_bars=0,
            params={"regular_minutes": self.regular_minutes},
        )

    def expression(self) -> pl.Expr:
        length = (pl.col("session_close") - pl.col("session_open")).dt.total_microseconds()
        return (
            pl.when(length.is_null())
            .then(None)
            .otherwise((length < self.regular_minutes * _MINUTE_US).cast(pl.Int8))
        )
