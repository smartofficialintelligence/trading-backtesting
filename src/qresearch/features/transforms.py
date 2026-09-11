"""Fitted transforms: feature preprocessing with fold-owned state.

The global-preprocessing leak (ARCHITECTURE.md sec. 9) is a scaler, imputer, or encoder
fitted on the whole sample and then applied to the training rows, so that training
"knows" the test distribution. The defence is structural:

* :meth:`Transform.fit` accepts only :class:`TrainingData`, and ``TrainingData`` can only
  be built from a split whose role is fittable (train). Handing it validation or test data
  is a construction error, not a silent mistake.
* Fitting returns a serialisable :class:`FittedState` that records what it was fitted on.
  The orchestrator persists one per fold; ``apply`` takes the state explicitly, so which
  fold's statistics are in use is always visible.
* ``apply`` is row-wise by contract. It may read fitted state and the row; never other
  rows. That keeps a transformed frame's ``available_at`` unchanged and lets the leakage
  checkers in :mod:`qresearch.features.leakage` verify a transform the same way as a
  feature.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import polars as pl
from pydantic import Field

from qresearch.config import FrozenModel
from qresearch.research.splits import DataSplit, TimeRange, select_split


@dataclass(frozen=True, slots=True)
class TrainingData:
    """A frame the orchestrator vouches for as training-role.

    Construct via :func:`training_data`; direct construction with a non-fittable split
    raises.
    """

    frame: pl.DataFrame
    split: DataSplit

    def __post_init__(self) -> None:
        if not self.split.fittable:
            raise ValueError(
                f"refusing to build training data from a {self.split.role.value!r} split; "
                "transforms may be fitted on training-role data only"
            )


def training_data(
    frame: pl.DataFrame, split: DataSplit, *, on: str = "available_at"
) -> TrainingData:
    """Select the rows of ``frame`` inside a training split and tag them as such."""
    return TrainingData(frame=select_split(frame, split, on=on), split=split)


class FittedState(FrozenModel):
    """Serialisable result of fitting one transform on one fold."""

    transform: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    fold: int = Field(ge=0)
    fitted_on: TimeRange
    row_count: int = Field(ge=0)
    columns: tuple[str, ...]
    statistics: dict[str, dict[str, float]]
    """Per-column fitted numbers, e.g. ``{"ret_1": {"mean": ..., "std": ...}}``."""


class Transform(Protocol):
    """A preprocessing step with fold-owned fitted state."""

    @property
    def name(self) -> str: ...

    def fit(self, data: TrainingData) -> FittedState: ...

    def apply(self, frame: pl.DataFrame, state: FittedState) -> pl.DataFrame: ...


def _scalar(value: object, *, what: str) -> float:
    """Narrow a Polars aggregate to a float; the stubs type them as broad unions."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{what}: expected a numeric statistic, got {type(value).__name__}")
    return float(value)


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], *, what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what}: columns {missing} are not present")


def _check_state(state: FittedState, *, name: str, columns: Sequence[str]) -> None:
    if state.transform != name:
        raise ValueError(f"state was fitted by {state.transform!r}, not {name!r}")
    if tuple(columns) != state.columns:
        raise ValueError(
            f"state was fitted on columns {state.columns}, transform targets {tuple(columns)}"
        )


@dataclass(frozen=True, slots=True)
class ZScoreScaler:
    """``(x - mean) / std`` with mean and std from the training split only.

    A column with zero training std is left null after scaling rather than divided by
    zero: a constant feature carries no information and an infinite one is not a value.
    """

    columns: tuple[str, ...]
    ddof: int = 1

    @property
    def name(self) -> str:
        return "zscore"

    def fit(self, data: TrainingData) -> FittedState:
        _require_columns(data.frame, self.columns, what="zscore fit")
        stats: dict[str, dict[str, float]] = {}
        for column in self.columns:
            series = data.frame.get_column(column).drop_nulls()
            if series.is_empty():
                raise ValueError(f"zscore fit: column {column!r} has no non-null training rows")
            mean = _scalar(series.mean(), what="zscore fit")
            std = _scalar(series.std(ddof=self.ddof) or 0.0, what="zscore fit")
            stats[column] = {"mean": mean, "std": std}
        return FittedState(
            transform=self.name,
            fold=data.split.fold,
            fitted_on=data.split.range,
            row_count=data.frame.height,
            columns=self.columns,
            statistics=stats,
        )

    def apply(self, frame: pl.DataFrame, state: FittedState) -> pl.DataFrame:
        _check_state(state, name=self.name, columns=self.columns)
        _require_columns(frame, self.columns, what="zscore apply")
        exprs = []
        for column in self.columns:
            mean, std = state.statistics[column]["mean"], state.statistics[column]["std"]
            scaled = (pl.col(column) - mean) / std if std > 0 else pl.lit(None, dtype=pl.Float64)
            exprs.append(scaled.alias(column))
        return frame.with_columns(*exprs)


@dataclass(frozen=True, slots=True)
class Winsorizer:
    """Clip each column to training-split quantiles ``[lower, upper]``."""

    columns: tuple[str, ...]
    lower: float = 0.01
    upper: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError(f"need 0 <= lower < upper <= 1, got {self.lower}, {self.upper}")

    @property
    def name(self) -> str:
        return "winsorize"

    def fit(self, data: TrainingData) -> FittedState:
        _require_columns(data.frame, self.columns, what="winsorize fit")
        stats: dict[str, dict[str, float]] = {}
        for column in self.columns:
            series = data.frame.get_column(column).drop_nulls()
            if series.is_empty():
                raise ValueError(f"winsorize fit: column {column!r} has no non-null training rows")
            lo = _scalar(series.quantile(self.lower, interpolation="linear"), what="winsorize fit")
            hi = _scalar(series.quantile(self.upper, interpolation="linear"), what="winsorize fit")
            stats[column] = {"lower": lo, "upper": hi}
        return FittedState(
            transform=self.name,
            fold=data.split.fold,
            fitted_on=data.split.range,
            row_count=data.frame.height,
            columns=self.columns,
            statistics=stats,
        )

    def apply(self, frame: pl.DataFrame, state: FittedState) -> pl.DataFrame:
        _check_state(state, name=self.name, columns=self.columns)
        _require_columns(frame, self.columns, what="winsorize apply")
        return frame.with_columns(
            *[
                pl.col(c).clip(state.statistics[c]["lower"], state.statistics[c]["upper"]).alias(c)
                for c in self.columns
            ]
        )
