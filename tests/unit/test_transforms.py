"""Fitted transforms: fold-owned state, and the refusal to fit on the wrong role."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.unit.test_resample import minute_bars

from qresearch.features.leakage import assert_future_insensitive, assert_prefix_invariant
from qresearch.features.pipeline import compute_features
from qresearch.features.technical import LaggedReturn, RelativeVolume
from qresearch.features.transforms import (
    FittedState,
    TrainingData,
    Winsorizer,
    ZScoreScaler,
    training_data,
)
from qresearch.research.splits import DataSplit, SplitRole, TimeRange

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)


def at(minute: int) -> dt.datetime:
    return T0 + dt.timedelta(minutes=minute)


def split(role: SplitRole, a: int, b: int, fold: int = 0) -> DataSplit:
    return DataSplit(role=role, range=TimeRange(start=at(a), end=at(b)), fold=fold)


@pytest.fixture
def features() -> pl.DataFrame:
    return compute_features(minute_bars(60), [LaggedReturn(1), RelativeVolume(3)])


# -- the refusal -------------------------------------------------------------------------


@pytest.mark.parametrize("role", [r for r in SplitRole if r is not SplitRole.TRAIN])
def test_training_data_refuses_every_non_train_role(
    features: pl.DataFrame, role: SplitRole
) -> None:
    with pytest.raises(ValueError, match="training-role data only"):
        training_data(features, split(role, 0, 30))
    with pytest.raises(ValueError, match="training-role data only"):
        TrainingData(frame=features, split=split(role, 0, 30))


def test_fit_signature_admits_only_training_data(features: pl.DataFrame) -> None:
    """A bare frame cannot be fitted on: the type is the guard."""
    scaler = ZScoreScaler(("ret_1",))
    with pytest.raises(AttributeError):
        scaler.fit(features)  # type: ignore[arg-type]


# -- fold-owned state --------------------------------------------------------------------


def test_state_records_what_it_was_fitted_on(features: pl.DataFrame) -> None:
    state = ZScoreScaler(("ret_1", "rvol_3")).fit(
        training_data(features, split(SplitRole.TRAIN, 0, 30, fold=2))
    )
    assert state.transform == "zscore"
    assert state.fold == 2
    assert state.fitted_on == TimeRange(start=at(0), end=at(30))
    assert state.columns == ("ret_1", "rvol_3")
    assert state.row_count == features.filter(pl.col("available_at") < at(30)).height
    assert set(state.statistics["ret_1"]) == {"mean", "std"}


def test_state_round_trips_through_json(features: pl.DataFrame) -> None:
    state = Winsorizer(("ret_1",)).fit(training_data(features, split(SplitRole.TRAIN, 0, 30)))
    assert FittedState.model_validate_json(state.model_dump_json()) == state


def test_statistics_come_from_the_training_rows_only(features: pl.DataFrame) -> None:
    """Fitting on [0, 30) must not know about rows available after 00:30."""
    train = training_data(features, split(SplitRole.TRAIN, 0, 30))
    state = ZScoreScaler(("ret_1",)).fit(train)
    expected_mean = train.frame.get_column("ret_1").drop_nulls().mean()
    global_mean = features.get_column("ret_1").drop_nulls().mean()
    assert state.statistics["ret_1"]["mean"] == pytest.approx(expected_mean)
    assert state.statistics["ret_1"]["mean"] != pytest.approx(global_mean)


def test_apply_uses_the_given_state_not_the_frame(features: pl.DataFrame) -> None:
    state = ZScoreScaler(("ret_1",)).fit(training_data(features, split(SplitRole.TRAIN, 0, 30)))
    test_rows = features.filter(pl.col("available_at") >= at(30))
    out = ZScoreScaler(("ret_1",)).apply(test_rows, state)
    mean, std = state.statistics["ret_1"]["mean"], state.statistics["ret_1"]["std"]
    expected = (test_rows.get_column("ret_1") - mean) / std
    assert out.get_column("ret_1").to_list() == pytest.approx(expected.to_list(), nan_ok=True)
    assert (
        out.get_column("available_at").to_list() == test_rows.get_column("available_at").to_list()
    )


def test_apply_refuses_a_state_from_another_transform_or_columns(features: pl.DataFrame) -> None:
    z = ZScoreScaler(("ret_1",)).fit(training_data(features, split(SplitRole.TRAIN, 0, 30)))
    with pytest.raises(ValueError, match="fitted by 'zscore'"):
        Winsorizer(("ret_1",)).apply(features, z)
    with pytest.raises(ValueError, match="fitted on columns"):
        ZScoreScaler(("rvol_3",)).apply(features, z)


def test_zero_variance_column_scales_to_null(features: pl.DataFrame) -> None:
    constant = features.with_columns(pl.lit(1.0).alias("ret_1"))
    scaler = ZScoreScaler(("ret_1",))
    state = scaler.fit(training_data(constant, split(SplitRole.TRAIN, 0, 30)))
    assert state.statistics["ret_1"]["std"] == 0.0
    assert scaler.apply(constant, state).get_column("ret_1").null_count() == constant.height


def test_winsorizer_clips_to_training_quantiles(features: pl.DataFrame) -> None:
    w = Winsorizer(("ret_1",), lower=0.1, upper=0.9)
    state = w.fit(training_data(features, split(SplitRole.TRAIN, 0, 30)))
    lo, hi = state.statistics["ret_1"]["lower"], state.statistics["ret_1"]["upper"]
    out = w.apply(features, state).get_column("ret_1").drop_nulls()
    assert out.min() >= lo and out.max() <= hi


def test_winsorizer_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        Winsorizer(("x",), lower=0.9, upper=0.1)


def test_fit_needs_non_null_rows(features: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="no non-null training rows"):
        ZScoreScaler(("ret_1",)).fit(training_data(features, split(SplitRole.TRAIN, 0, 1)))


# -- leakage --------------------------------------------------------------------------


def test_a_fitted_transform_passes_the_leakage_checks_with_fixed_state() -> None:
    """With state fixed, applying is row-wise: no dependence on other rows."""
    bars = minute_bars(80, late={9: dt.timedelta(minutes=6)}, skip={40})
    base = compute_features(bars, [LaggedReturn(1)])
    state = ZScoreScaler(("ret_1",)).fit(training_data(base, split(SplitRole.TRAIN, 0, 40)))

    def compute(b: pl.DataFrame) -> pl.DataFrame:
        return ZScoreScaler(("ret_1",)).apply(compute_features(b, [LaggedReturn(1)]), state)

    assert_prefix_invariant(compute, bars)
    assert_future_insensitive(compute, bars)


def test_refitting_inside_compute_is_the_global_preprocessing_leak() -> None:
    """The canonical mistake, demonstrated: fitting on whatever frame is at hand."""
    from qresearch.features.leakage import LeakageDetected

    bars = minute_bars(80)

    def leaky(b: pl.DataFrame) -> pl.DataFrame:
        base = compute_features(b, [LaggedReturn(1)])
        whole = split(SplitRole.TRAIN, 0, 10_000)  # "train" on everything present
        state = ZScoreScaler(("ret_1",)).fit(training_data(base, whole))
        return ZScoreScaler(("ret_1",)).apply(base, state)

    with pytest.raises(LeakageDetected):
        assert_prefix_invariant(leaky, bars)
