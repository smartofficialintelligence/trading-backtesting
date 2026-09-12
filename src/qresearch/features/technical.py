"""Price- and volume-derived features.

Each feature is a small frozen dataclass exposing a :class:`FeatureSpec` and one Polars
expression. Every feature here passes the prefix-invariance and future-insensitivity
checks in :mod:`qresearch.features.leakage`; a new feature should be added to that test
matrix before it is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import polars as pl

from qresearch.features.contracts import FeatureSpec


def _qualname(obj: object) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True, slots=True)
class LaggedReturn:
    """Return of the current close over the close ``lag`` bars earlier.

    ``ret_1`` at bar N is ``close[N] / close[N-1] - 1``: the return *of* bar N, known once
    bar N is complete. It is the most basic causal feature and the reference example for
    the timing contract -- its value at N depends on bars N-lag..N and becomes available
    when the last of those does.
    """

    lag: int = 1
    kind: Literal["simple", "log"] = "simple"
    require_contiguous: bool = True

    def __post_init__(self) -> None:
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")

    @property
    def spec(self) -> FeatureSpec:
        suffix = "" if self.kind == "simple" else "_log"
        return FeatureSpec(
            name=f"ret{suffix}_{self.lag}",
            implementation=_qualname(self),
            inputs=("close",),
            lookback_bars=self.lag,
            params={"lag": self.lag, "kind": self.kind},
            require_contiguous=self.require_contiguous,
        )

    def expression(self) -> pl.Expr:
        close = pl.col("close")
        previous = close.shift(self.lag)
        if self.kind == "log":
            return (close / previous).log()
        return close / previous - 1.0


@dataclass(frozen=True, slots=True)
class RollingVolatility:
    """Sample standard deviation of 1-bar returns over the previous ``window`` returns.

    ``vol_5`` at bar N uses the five returns ending at N, which need closes N-5..N. Not
    annualised: annualisation is a reporting decision with declared session assumptions
    (ARCHITECTURE.md sec. 8), not a property of the feature.
    """

    window: int = 20
    kind: Literal["simple", "log"] = "simple"
    require_contiguous: bool = True

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be >= 2 to form a standard deviation, got {self.window}")

    @property
    def spec(self) -> FeatureSpec:
        suffix = "" if self.kind == "simple" else "_log"
        return FeatureSpec(
            name=f"vol{suffix}_{self.window}",
            implementation=_qualname(self),
            inputs=("close",),
            lookback_bars=self.window,
            params={"window": self.window, "kind": self.kind},
            require_contiguous=self.require_contiguous,
        )

    def expression(self) -> pl.Expr:
        close = pl.col("close")
        ret = (close / close.shift(1)).log() if self.kind == "log" else close / close.shift(1) - 1.0
        return ret.rolling_std(window_size=self.window, min_samples=self.window)


@dataclass(frozen=True, slots=True)
class RelativeVolume:
    """Current bar's volume relative to the mean of the previous ``window`` bars.

    The trailing mean excludes the current bar, so ``rvol_20 = 2.0`` reads as "twice the
    recent norm". Null where the trailing mean is zero: a stretch of untraded bars gives
    no norm to be relative to, and an infinite ratio is not a feature value.
    """

    window: int = 20
    require_contiguous: bool = True

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"rvol_{self.window}",
            implementation=_qualname(self),
            inputs=("volume",),
            lookback_bars=self.window,
            params={"window": self.window},
            require_contiguous=self.require_contiguous,
        )

    def expression(self) -> pl.Expr:
        trailing = (
            pl.col("volume").shift(1).rolling_mean(window_size=self.window, min_samples=self.window)
        )
        return pl.when(trailing > 0).then(pl.col("volume") / trailing).otherwise(None)


@dataclass(frozen=True, slots=True)
class BarRange:
    """``(high - low) / close`` of the current bar: a same-bar, zero-lookback feature."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="bar_range",
            implementation=_qualname(self),
            inputs=("high", "low", "close"),
            lookback_bars=0,
        )

    def expression(self) -> pl.Expr:
        return (pl.col("high") - pl.col("low")) / pl.col("close")


@dataclass(frozen=True, slots=True)
class MinuteOfDay:
    """Minute of the UTC day of ``bar_start`` (0..1439), optionally cyclically encoded.

    UTC deliberately: a session-relative clock ("minutes since the open") needs an
    exchange calendar and belongs to the session features that depend on one. For 24/7
    markets this is the natural clock; for equities it is still causal, just less
    meaningful than a session clock.
    """

    encoding: Literal["raw", "sin", "cos"] = "raw"

    @property
    def spec(self) -> FeatureSpec:
        suffix = "" if self.encoding == "raw" else f"_{self.encoding}"
        return FeatureSpec(
            name=f"minute_of_day_utc{suffix}",
            implementation=_qualname(self),
            inputs=("bar_start",),
            lookback_bars=0,
            params={"encoding": self.encoding},
        )

    def expression(self) -> pl.Expr:
        # dt.hour()/dt.minute() are Int8; 13 * 60 would wrap. Widen first.
        start = pl.col("bar_start")
        minute = start.dt.hour().cast(pl.Int32) * 60 + start.dt.minute().cast(pl.Int32)
        if self.encoding == "raw":
            return minute
        angle = minute.cast(pl.Float64) * (2.0 * math.pi / 1440.0)
        return angle.sin() if self.encoding == "sin" else angle.cos()


@dataclass(frozen=True, slots=True)
class DayOfWeek:
    """Day of the UTC week of ``bar_start``: 0 = Monday .. 6 = Sunday."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="day_of_week_utc",
            implementation=_qualname(self),
            inputs=("bar_start",),
            lookback_bars=0,
        )

    def expression(self) -> pl.Expr:
        # Polars weekday() is ISO: Monday = 1 .. Sunday = 7.
        return pl.col("bar_start").dt.weekday().cast(pl.Int32) - 1


@dataclass(frozen=True, slots=True)
class RollingZScore:
    """How far the current price sits from its recent mean, in standard deviations.

    ``(close - mean(window)) / std(window)``, where the window ends at and includes the
    current bar. This is the canonical mean-reversion signal: a large negative value says
    the price is stretched below its own recent average.

    It measures *stretch*, not direction of travel, and says nothing about whether the
    mean is itself moving. In a trend it is persistently signed and reverts slowly or not
    at all -- which is the honest reason mean-reversion strategies need a regime filter
    rather than a stretch threshold alone.

    Null where the window's standard deviation is zero: a flat window has no scale to
    measure against, and an infinite z-score is not a feature value.
    """

    window: int = 20
    column: str = "close"
    require_contiguous: bool = True

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be >= 2 to form a standard deviation, got {self.window}")

    @property
    def spec(self) -> FeatureSpec:
        suffix = "" if self.column == "close" else f"_{self.column}"
        return FeatureSpec(
            name=f"zscore{suffix}_{self.window}",
            implementation=_qualname(self),
            inputs=(self.column,),
            # The window includes the current bar, so it reaches back window - 1 bars.
            lookback_bars=self.window - 1,
            params={"window": self.window, "column": self.column},
            require_contiguous=self.require_contiguous,
        )

    def expression(self) -> pl.Expr:
        value = pl.col(self.column)
        mean = value.rolling_mean(window_size=self.window, min_samples=self.window)
        std = value.rolling_std(window_size=self.window, min_samples=self.window)
        return pl.when(std > 0).then((value - mean) / std).otherwise(None)


@dataclass(frozen=True, slots=True)
class RollingRange:
    """High-to-low range over the window, as a fraction of the latest close.

    A cheaper, more robust measure of variability than return volatility: it responds to a
    single violent bar rather than needing several, and is not thrown by a run of
    identical closes. Useful as the selection signal in "trade the most variable name".
    """

    window: int = 20
    require_contiguous: bool = True

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"range_{self.window}",
            implementation=_qualname(self),
            inputs=("high", "low", "close"),
            lookback_bars=self.window - 1,
            params={"window": self.window},
            require_contiguous=self.require_contiguous,
        )

    def expression(self) -> pl.Expr:
        high = pl.col("high").rolling_max(window_size=self.window, min_samples=self.window)
        low = pl.col("low").rolling_min(window_size=self.window, min_samples=self.window)
        return pl.when(pl.col("close") > 0).then((high - low) / pl.col("close")).otherwise(None)
