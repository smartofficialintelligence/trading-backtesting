"""The leakage checkers must pass honest features and fail dishonest ones.

A guard that has never been shown to fire is decoration. Each leaky computation below is
a realistic mistake, written to the same output shape as the pipeline, and each must be
caught by at least one checker -- the table at the bottom says which.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import polars as pl
import pytest
from tests.unit.test_resample import minute_bars

from qresearch.features.leakage import (
    LeakageDetected,
    assert_future_insensitive,
    assert_prefix_invariant,
    default_cut_points,
)
from qresearch.features.pipeline import compute_features
from qresearch.features.technical import (
    BarRange,
    DayOfWeek,
    LaggedReturn,
    MinuteOfDay,
    RelativeVolume,
    RollingVolatility,
)

KEY = ["instrument_id", "bar_start"]
Compute = Callable[[pl.DataFrame], pl.DataFrame]


def pipeline(*features: object) -> Compute:
    return lambda bars: compute_features(bars, list(features))


def _shape(frame: pl.DataFrame, value: pl.Expr, availability: pl.Expr) -> pl.DataFrame:
    return (
        frame.sort(KEY)
        .with_columns(value.over("instrument_id").alias("f"), availability.alias("available_at"))
        .select(*KEY, "f", "available_at")
    )


# -- honest ----------------------------------------------------------------------------

DATASETS = {
    "clean": minute_bars(40),
    "with_gap": minute_bars(40, skip={12, 13}),
    "with_late_bar": minute_bars(40, late={9: dt.timedelta(minutes=6)}),
    "gap_and_late": minute_bars(
        40, skip={20}, late={9: dt.timedelta(minutes=6), 30: dt.timedelta(minutes=3)}
    ),
}
HONEST = {
    "ret_1": pipeline(LaggedReturn(1)),
    "ret_3": pipeline(LaggedReturn(3)),
    "ret_log_2": pipeline(LaggedReturn(2, kind="log")),
    "ret_1_and_5": pipeline(LaggedReturn(1), LaggedReturn(5)),
    "ret_2_gap_tolerant": pipeline(LaggedReturn(2, require_contiguous=False)),
    "vol_5": pipeline(RollingVolatility(5)),
    "vol_log_3_gap_tolerant": pipeline(RollingVolatility(3, kind="log", require_contiguous=False)),
    "rvol_4": pipeline(RollingVolatility(2), RelativeVolume(4)),
    "bar_range": pipeline(BarRange()),
    "time": pipeline(MinuteOfDay(), MinuteOfDay("sin"), DayOfWeek()),
    "everything": pipeline(
        LaggedReturn(1),
        LaggedReturn(5),
        RollingVolatility(10),
        RelativeVolume(3),
        BarRange(),
        MinuteOfDay("cos"),
        DayOfWeek(),
    ),
}


@pytest.mark.parametrize("feature", list(HONEST))
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_honest_features_pass_both_checks(feature: str, dataset: str) -> None:
    compute, bars = HONEST[feature], DATASETS[dataset]
    assert_prefix_invariant(compute, bars)
    assert_future_insensitive(compute, bars)


# -- dishonest -------------------------------------------------------------------------


def centred_mean(bars: pl.DataFrame) -> pl.DataFrame:
    """A 3-bar centred moving average: reads the next bar."""
    return _shape(bars, pl.col("close").rolling_mean(3, center=True), pl.col("available_at"))


def next_bar_return(bars: pl.DataFrame) -> pl.DataFrame:
    """The classic label-as-feature mistake: shift(-1)."""
    return _shape(bars, pl.col("close").shift(-1) / pl.col("close") - 1, pl.col("available_at"))


def global_zscore(bars: pl.DataFrame) -> pl.DataFrame:
    """Normalised with the whole sample's mean and std."""
    z = (pl.col("close") - pl.col("close").mean()) / pl.col("close").std()
    return _shape(bars, z, pl.col("available_at"))


def lag_return_with_own_availability(bars: pl.DataFrame) -> pl.DataFrame:
    """Correct arithmetic, wrong timing: availability ignores the lagged input.

    This is the subtle one. The value is a perfectly causal lag-1 return, but the row
    claims to be usable when its *own* bar arrives -- not when the previous bar it reads
    has arrived. On an in-order dataset it is indistinguishable from the honest feature;
    only a late bar exposes it.
    """
    return _shape(bars, pl.col("close") / pl.col("close").shift(1) - 1, pl.col("available_at"))


def overly_conservative_availability(bars: pl.DataFrame) -> pl.DataFrame:
    """Causal value (``cum_max`` looks only backward) stamped with the *dataset's* last
    availability, so every row claims to be usable only at the very end.

    Not a leak. The checkers test whether a row is usable *earlier* than its inputs
    allow; a row that is usable later than necessary is wasteful but honest, and both
    checks pass it (vacuously -- nothing is ever available before an interior cut). It is
    in the matrix to pin down that boundary: these helpers detect look-ahead, not
    uselessness.
    """
    return _shape(
        bars, pl.col("close") / pl.col("close").cum_max() - 1, pl.col("available_at").max()
    )


@pytest.mark.parametrize(
    ("leaky", "dataset", "prefix_fails", "future_fails"),
    [
        (centred_mean, "clean", True, True),
        (next_bar_return, "clean", True, True),
        (global_zscore, "clean", True, True),
        (lag_return_with_own_availability, "clean", False, False),
        (lag_return_with_own_availability, "with_late_bar", True, True),
        (overly_conservative_availability, "clean", False, False),
    ],
    ids=lambda v: v if isinstance(v, str) else getattr(v, "__name__", str(v)),
)
def test_dishonest_features_are_caught(
    leaky: Compute, dataset: str, prefix_fails: bool, future_fails: bool
) -> None:
    bars = DATASETS[dataset]
    if prefix_fails:
        with pytest.raises(LeakageDetected):
            assert_prefix_invariant(leaky, bars)
    else:
        assert_prefix_invariant(leaky, bars)
    if future_fails:
        with pytest.raises(LeakageDetected):
            assert_future_insensitive(leaky, bars)
    else:
        assert_future_insensitive(leaky, bars)


def test_the_subtle_case_needs_a_late_bar_to_surface() -> None:
    """Documents why the test matrix includes out-of-order availability: on in-order data
    the wrong-availability feature is invisible to both checks."""
    assert_prefix_invariant(lag_return_with_own_availability, DATASETS["clean"])
    with pytest.raises(LeakageDetected):
        assert_prefix_invariant(lag_return_with_own_availability, DATASETS["with_late_bar"])


def test_failures_carry_the_offending_rows() -> None:
    with pytest.raises(LeakageDetected) as caught:
        assert_future_insensitive(next_bar_return, DATASETS["clean"])
    assert isinstance(caught.value, AssertionError)
    assert caught.value.offending.height > 0
    assert "f" in caught.value.offending.columns
    assert "f__corrupt" in caught.value.offending.columns


# -- helper hygiene ----------------------------------------------------------------------


def test_cut_points_are_interior_and_sorted() -> None:
    bars = minute_bars(20)
    cuts = default_cut_points(bars, count=4)
    series = bars.get_column("available_at")
    assert list(cuts) == sorted(cuts)
    assert series.min() < cuts[0] and cuts[-1] < series.max()


def test_cut_points_need_at_least_three_bars() -> None:
    with pytest.raises(ValueError, match="at least three"):
        default_cut_points(minute_bars(2))


def test_explicit_cut_points_are_honoured() -> None:
    bars = minute_bars(10)
    cut = bars.item(4, "available_at")
    assert_prefix_invariant(pipeline(LaggedReturn(1)), bars, cut_points=[cut])


def test_a_compute_with_the_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="must return a frame with"):
        assert_prefix_invariant(lambda b: b.select("close"), minute_bars(5))


def test_nan_equal_to_nan_does_not_count_as_a_difference() -> None:
    """A feature that legitimately yields NaN must not trip the checker on NaN != NaN."""

    def nan_feature(bars: pl.DataFrame) -> pl.DataFrame:
        return _shape(bars, pl.lit(float("nan")), pl.col("available_at"))

    assert_prefix_invariant(nan_feature, minute_bars(10))
    assert_future_insensitive(nan_feature, minute_bars(10))


# -- tolerance -----------------------------------------------------------------------------


def _length_dependent(scale: float) -> Compute:
    """Causal value nudged by a term that depends on the frame's length -- a global
    dependence, but one of adjustable size."""

    def compute(bars: pl.DataFrame) -> pl.DataFrame:
        return _shape(bars, pl.col("close") * (1.0 + scale * pl.len()), pl.col("available_at"))

    return compute


def test_last_bit_differences_are_tolerated_by_default() -> None:
    """Streaming rolling statistics are not bit-reproducible across prefixes; the
    checkers must not fail an honest feature over rounding noise."""
    assert_prefix_invariant(_length_dependent(1e-14), minute_bars(40))
    assert_future_insensitive(_length_dependent(1e-14), minute_bars(40))


def test_material_differences_are_not() -> None:
    with pytest.raises(LeakageDetected):
        assert_prefix_invariant(_length_dependent(1e-6), minute_bars(40))


def test_rtol_zero_demands_exact_reproduction() -> None:
    with pytest.raises(LeakageDetected):
        assert_prefix_invariant(_length_dependent(1e-14), minute_bars(40), rtol=0)


def test_negative_rtol_is_refused() -> None:
    with pytest.raises(ValueError, match="rtol must be >= 0"):
        assert_prefix_invariant(pipeline(LaggedReturn(1)), minute_bars(10), rtol=-1.0)
