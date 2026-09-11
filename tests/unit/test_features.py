"""Feature pipeline: values, warm-up, contiguity, and -- above all -- availability."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.unit.test_resample import minute_bars

from qresearch.features.contracts import FeatureSpec
from qresearch.features.pipeline import compute_features
from qresearch.features.technical import (
    BarRange,
    DayOfWeek,
    LaggedReturn,
    MinuteOfDay,
    RelativeVolume,
    RollingVolatility,
)

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)


def at(minute: int, second: int = 0) -> dt.datetime:
    return T0 + dt.timedelta(minutes=minute, seconds=second)


def close(i: int) -> float:
    """Matches ``minute_bars``: close = 100.2 + i."""
    return 100.2 + i


# -- values --------------------------------------------------------------------------


def test_lag_one_return_matches_hand_calculation() -> None:
    out = compute_features(minute_bars(4), [LaggedReturn(1)])
    values = out.get_column("ret_1").to_list()
    assert values[0] is None, "no prior bar: warm-up"
    for i in range(1, 4):
        assert values[i] == pytest.approx(close(i) / close(i - 1) - 1)


def test_lag_three_return_warms_up_for_three_bars() -> None:
    out = compute_features(minute_bars(6), [LaggedReturn(3)])
    values = out.get_column("ret_3").to_list()
    assert values[:3] == [None, None, None]
    assert values[3] == pytest.approx(close(3) / close(0) - 1)


def test_log_return() -> None:
    import math

    out = compute_features(minute_bars(3), [LaggedReturn(1, kind="log")])
    assert "ret_log_1" in out.columns
    assert out.item(1, "ret_log_1") == pytest.approx(math.log(close(1) / close(0)))


def test_output_shape_and_order() -> None:
    out = compute_features(minute_bars(3), [LaggedReturn(1), LaggedReturn(2)])
    assert out.columns == ["instrument_id", "bar_start", "ret_1", "ret_2", "available_at"]
    assert out.get_column("bar_start").is_sorted()


# -- contiguity ----------------------------------------------------------------------


def test_a_gap_nulls_the_window_that_spans_it() -> None:
    """Bar 3 is missing. ret_1 at bar 4 would be close(4)/close(2): a two-minute return
    wearing a one-minute label. It must be null."""
    out = compute_features(minute_bars(6, skip={3}), [LaggedReturn(1)])
    by_start = dict(zip(out.get_column("bar_start"), out.get_column("ret_1"), strict=True))
    assert by_start[at(2)] is not None
    assert by_start[at(4)] is None
    assert by_start[at(5)] is not None, "the window {4, 5} is contiguous again"


def test_a_gap_nulls_every_window_it_falls_inside() -> None:
    out = compute_features(minute_bars(8, skip={3}), [LaggedReturn(3)])
    by_start = dict(zip(out.get_column("bar_start"), out.get_column("ret_3"), strict=True))
    # Windows ending at 4, 5, 6 all include the missing bar 3; 7's window is {4,5,6,7}.
    assert by_start[at(4)] is None
    assert by_start[at(6)] is None
    assert by_start[at(7)] is not None


def test_contiguity_can_be_opted_out() -> None:
    out = compute_features(minute_bars(6, skip={3}), [LaggedReturn(1, require_contiguous=False)])
    by_start = dict(zip(out.get_column("bar_start"), out.get_column("ret_1"), strict=True))
    assert by_start[at(4)] == pytest.approx(close(4) / close(2) - 1)


# -- availability ----------------------------------------------------------------------


def test_availability_is_the_bars_own_when_inputs_arrive_in_order() -> None:
    bars = minute_bars(5)
    out = compute_features(bars, [LaggedReturn(1)])
    assert out.get_column("available_at").to_list() == bars.get_column("available_at").to_list()


def test_a_late_input_delays_every_feature_that_reads_it() -> None:
    """Bar 5 publishes 9 minutes late (at 00:15). ret_1 at bar 6 reads close(5), so it
    cannot be available before 00:15 even though bar 6 itself arrived at 00:07:02."""
    out = compute_features(minute_bars(9, late={5: dt.timedelta(minutes=9)}), [LaggedReturn(1)])
    avail = dict(zip(out.get_column("bar_start"), out.get_column("available_at"), strict=True))
    assert avail[at(5)] == at(6) + dt.timedelta(minutes=9)
    assert avail[at(6)] == at(6) + dt.timedelta(minutes=9), "window {5, 6} waits for 5"
    assert avail[at(7)] == at(8, 2), "window {6, 7} does not include the late bar"


def test_a_late_input_delays_a_longer_window_for_longer() -> None:
    out = compute_features(minute_bars(12, late={5: dt.timedelta(minutes=9)}), [LaggedReturn(3)])
    avail = dict(zip(out.get_column("bar_start"), out.get_column("available_at"), strict=True))
    late = at(6) + dt.timedelta(minutes=9)
    assert all(avail[at(i)] == late for i in range(5, 9)), "windows ending 5..8 include bar 5"
    assert avail[at(9)] == at(10, 2)


def test_row_availability_is_the_max_across_features() -> None:
    bars = minute_bars(12, late={5: dt.timedelta(minutes=9)})
    one = compute_features(bars, [LaggedReturn(1)])
    three = compute_features(bars, [LaggedReturn(3)])
    both = compute_features(bars, [LaggedReturn(1), LaggedReturn(3)])
    expected = [
        max(a, b)
        for a, b in zip(
            one.get_column("available_at").to_list(),
            three.get_column("available_at").to_list(),
            strict=True,
        )
    ]
    assert both.get_column("available_at").to_list() == expected


def test_computation_latency_is_added() -> None:
    class Slow(LaggedReturn):
        __slots__ = ()

        @property
        def spec(self) -> FeatureSpec:
            return super().spec.model_copy(update={"computation_latency": dt.timedelta(seconds=30)})

    bars = minute_bars(4)
    out = compute_features(bars, [Slow(1)])
    delayed = out.get_column("available_at").to_list()
    original = bars.get_column("available_at").to_list()
    assert delayed[0] == original[0], "warm-up row: no window yet, falls back to the bar"
    assert delayed[1] == original[1] + dt.timedelta(seconds=30)


def test_warm_up_rows_are_available_when_their_bar_is() -> None:
    """A null feature during warm-up is an honest 'not enough history yet' at that time,
    not something that should be hidden until history exists."""
    bars = minute_bars(3)
    out = compute_features(bars, [LaggedReturn(2)])
    assert out.get_column("ret_2").to_list()[:2] == [None, None]
    assert (
        out.get_column("available_at").to_list()[:2]
        == bars.get_column("available_at").to_list()[:2]
    )


# -- multiple instruments ----------------------------------------------------------------


def test_instruments_are_independent() -> None:
    a = minute_bars(4)
    b = minute_bars(4).with_columns(
        pl.lit("Y").alias("instrument_id"), (pl.col("close") * 2).alias("close")
    )
    out = compute_features(pl.concat([a, b]), [LaggedReturn(1)])
    ya = out.filter(pl.col("instrument_id") == "X").get_column("ret_1").to_list()
    yb = out.filter(pl.col("instrument_id") == "Y").get_column("ret_1").to_list()
    assert ya == pytest.approx(yb), "doubling every close leaves returns unchanged"
    assert ya[0] is None and yb[0] is None, "each instrument warms up separately"


def test_input_row_order_does_not_change_the_result() -> None:
    a = minute_bars(6)
    b = minute_bars(6).with_columns(pl.lit("Y").alias("instrument_id"))
    frame = pl.concat([a, b])
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=7)
    assert compute_features(frame, [LaggedReturn(2)]).equals(
        compute_features(shuffled, [LaggedReturn(2)])
    )


# -- validation ------------------------------------------------------------------------


def test_duplicate_feature_names_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate feature names"):
        compute_features(minute_bars(3), [LaggedReturn(1), LaggedReturn(1)])


def test_missing_input_columns_are_refused() -> None:
    with pytest.raises(ValueError, match="require bar columns \\['close'\\]"):
        compute_features(minute_bars(3).drop("close"), [LaggedReturn(1)])


def test_mixed_bar_sizes_are_refused() -> None:
    mixed = pl.concat([minute_bars(3), minute_bars(3).with_columns(pl.lit("5m").alias("bar_size"))])
    with pytest.raises(ValueError, match="exactly one bar_size"):
        compute_features(mixed, [LaggedReturn(1)])


def test_empty_frame_and_empty_feature_list_are_refused() -> None:
    with pytest.raises(ValueError, match="empty bar frame"):
        compute_features(minute_bars(0), [LaggedReturn(1)])
    with pytest.raises(ValueError, match="no features"):
        compute_features(minute_bars(3), [])


def test_lag_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lag must be >= 1"):
        LaggedReturn(0)


# -- spec ------------------------------------------------------------------------------


def test_spec_fingerprint_identifies_parameters_and_implementation() -> None:
    assert LaggedReturn(1).spec.fingerprint == LaggedReturn(1).spec.fingerprint
    assert LaggedReturn(1).spec.fingerprint != LaggedReturn(2).spec.fingerprint
    assert LaggedReturn(1).spec.fingerprint != LaggedReturn(1, kind="log").spec.fingerprint
    assert LaggedReturn(1).spec.implementation.endswith("technical.LaggedReturn")


def test_spec_names_must_be_sql_and_python_safe() -> None:
    with pytest.raises(ValueError):
        FeatureSpec(name="Ret-1", implementation="x", inputs=("close",), lookback_bars=1)


# -- rolling volatility -------------------------------------------------------------------


def test_rolling_volatility_matches_sample_stdev_of_returns() -> None:
    from statistics import stdev

    out = compute_features(minute_bars(6), [RollingVolatility(3)])
    values = out.get_column("vol_3").to_list()
    assert values[:3] == [None, None, None], "three returns need four closes"
    returns = [close(i) / close(i - 1) - 1 for i in (1, 2, 3)]
    assert values[3] == pytest.approx(stdev(returns))


def test_rolling_volatility_spans_a_gap_only_if_allowed() -> None:
    strict = compute_features(minute_bars(10, skip={4}), [RollingVolatility(3)])
    loose = compute_features(
        minute_bars(10, skip={4}), [RollingVolatility(3, require_contiguous=False)]
    )
    by = lambda f: dict(zip(f.get_column("bar_start"), f.get_column("vol_3"), strict=True))  # noqa: E731
    assert by(strict)[at(6)] is None, "window {3,5,6} crosses the gap at 4"
    assert by(loose)[at(6)] is not None


def test_rolling_volatility_needs_window_of_at_least_two() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        RollingVolatility(1)


# -- relative volume ----------------------------------------------------------------------


def test_relative_volume_excludes_the_current_bar_from_the_norm() -> None:
    out = compute_features(minute_bars(6), [RelativeVolume(2)])
    values = out.get_column("rvol_2").to_list()
    assert values[:2] == [None, None]
    # volume = 10 + i; at bar 2 the trailing mean is (10 + 11) / 2.
    assert values[2] == pytest.approx(12 / 10.5)


def test_relative_volume_is_null_over_a_zero_norm() -> None:
    bars = minute_bars(5).with_columns(
        pl.when(pl.col("bar_start") < at(2)).then(0.0).otherwise(pl.col("volume")).alias("volume")
    )
    out = compute_features(bars, [RelativeVolume(2)])
    assert out.item(2, "rvol_2") is None, "trailing bars 0 and 1 have zero volume"
    assert out.item(4, "rvol_2") is not None


# -- zero-lookback features -----------------------------------------------------------------


def test_bar_range_is_available_with_its_own_bar_and_has_no_warm_up() -> None:
    bars = minute_bars(3)
    out = compute_features(bars, [BarRange()])
    assert out.get_column("bar_range").null_count() == 0
    assert out.item(0, "bar_range") == pytest.approx((100.5 - 99.5) / 100.2)
    assert out.get_column("available_at").to_list() == bars.get_column("available_at").to_list()


def test_zero_lookback_features_ignore_gaps() -> None:
    out = compute_features(minute_bars(5, skip={2}), [BarRange()])
    assert out.get_column("bar_range").null_count() == 0


# -- time features --------------------------------------------------------------------------


def test_minute_of_day_and_day_of_week_are_utc() -> None:
    bars = minute_bars(3).with_columns(
        (pl.col("bar_start") + pl.duration(hours=13, minutes=30)).alias("bar_start"),
        (pl.col("bar_end") + pl.duration(hours=13, minutes=30)).alias("bar_end"),
        (pl.col("available_at") + pl.duration(hours=13, minutes=30)).alias("available_at"),
    )
    out = compute_features(bars, [MinuteOfDay(), DayOfWeek()])
    assert out.get_column("minute_of_day_utc").to_list() == [810, 811, 812]
    assert out.get_column("minute_of_day_utc").dtype.is_integer()
    assert out.get_column("day_of_week_utc").to_list() == [0, 0, 0], "2024-03-04 is a Monday"


def test_cyclical_encodings_wrap_at_midnight() -> None:
    import math

    late = minute_bars(1).with_columns(
        (pl.col("bar_start") + pl.duration(minutes=1439)).alias("bar_start"),
        (pl.col("bar_end") + pl.duration(minutes=1439)).alias("bar_end"),
        (pl.col("available_at") + pl.duration(minutes=1439)).alias("available_at"),
    )
    frame = pl.concat([minute_bars(1), late])
    out = compute_features(frame, [MinuteOfDay("sin"), MinuteOfDay("cos")])
    sin, cos = out.get_column("minute_of_day_utc_sin"), out.get_column("minute_of_day_utc_cos")
    assert sin[0] == pytest.approx(0.0) and cos[0] == pytest.approx(1.0)
    assert sin[1] == pytest.approx(math.sin(2 * math.pi * 1439 / 1440))
    assert abs(cos[1] - cos[0]) < 1e-4, "23:59 is next to 00:00, not far from it"
