"""Registries resolve kinds and coerce parameters through dataclass annotations."""

from __future__ import annotations

import datetime as dt

import pytest

from qresearch.features.registry import build_feature, build_transform
from qresearch.features.technical import LaggedReturn, RollingVolatility
from qresearch.strategy.examples import ScheduledRebalance
from qresearch.strategy.registry import build_strategy


def test_features_by_kind() -> None:
    assert build_feature("lagged_return", {"lag": 3}) == LaggedReturn(3)
    assert build_feature("rolling_volatility", {"window": 5, "kind": "log"}) == RollingVolatility(
        5, kind="log"
    )
    assert build_feature("bar_range", {}).spec.name == "bar_range"


def test_parameters_are_coerced_through_annotations() -> None:
    assert build_feature("lagged_return", {"lag": "3"}) == LaggedReturn(3)
    strategy = build_strategy("scheduled_rebalance", {"weights": {"X": 0.5}, "every": "PT30M"})
    assert isinstance(strategy, ScheduledRebalance)
    assert strategy.every == dt.timedelta(minutes=30)


def test_unknown_kind_and_unknown_parameter_are_errors() -> None:
    with pytest.raises(KeyError, match="unknown feature kind"):
        build_feature("moving_average", {})
    with pytest.raises(ValueError, match="unknown parameters \\['lags'\\]"):
        build_feature("lagged_return", {"lags": 3})
    with pytest.raises(KeyError, match="unknown strategy kind"):
        build_strategy("hodl", {})


def test_transforms_take_columns_plus_params() -> None:
    transform = build_transform("winsorize", ("ret_1",), {"lower": 0.05, "upper": 0.95})
    assert transform.name == "winsorize"
    assert build_transform("zscore", ("a", "b"), {}).columns == ("a", "b")


def test_invalid_parameter_values_are_rejected_by_the_class() -> None:
    with pytest.raises(ValueError, match="lag must be >= 1"):
        build_feature("lagged_return", {"lag": 0})
