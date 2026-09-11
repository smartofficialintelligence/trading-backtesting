"""Cross-sectional ranks: batch availability and the missing-member policy."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.unit.test_resample import minute_bars

from qresearch.features.cross_sectional import CrossSectionalRank, compute_cross_sectional
from qresearch.features.leakage import assert_future_insensitive, assert_prefix_invariant
from qresearch.features.pipeline import compute_features
from qresearch.features.technical import LaggedReturn

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)


def at(minute: int, second: int = 0) -> dt.datetime:
    return T0 + dt.timedelta(minutes=minute, seconds=second)


def multi_bars(
    count: int,
    *,
    late: dict[str, dict[int, dt.timedelta]] | None = None,
    skip: dict[str, set[int]] | None = None,
    scale: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Three instruments A, B, C from ``minute_bars``, each with its own anomalies.

    ``scale`` multiplies closes so returns differ across instruments and ranks are
    non-trivial; the default gives A < B < C in every 1-bar return.
    """
    late, skip = late or {}, skip or {}
    scale = scale or {"A": 1.0, "B": 1.5, "C": 3.0}
    frames = []
    for name, factor in scale.items():
        base = minute_bars(count, late=late.get(name), skip=skip.get(name))
        frames.append(
            base.with_columns(
                pl.lit(name).alias("instrument_id"),
                # close = 100.2 + i for A; scaling the *increment* makes returns differ.
                (100.2 + (pl.col("close") - 100.2) * factor).alias("close"),
            )
        )
    return pl.concat(frames)


def ret_then_rank(bars: pl.DataFrame, **kwargs: object) -> pl.DataFrame:
    frame = compute_features(bars, [LaggedReturn(1)])
    return compute_cross_sectional(frame, [CrossSectionalRank("ret_1", **kwargs)])  # type: ignore[arg-type]


def column_at(frame: pl.DataFrame, column: str, minute: int) -> dict[str, object]:
    rows = frame.filter(pl.col("bar_start") == at(minute))
    return dict(zip(rows.get_column("instrument_id"), rows.get_column(column), strict=True))


# -- values --------------------------------------------------------------------------


def test_percentile_rank_orders_instruments_by_value() -> None:
    out = ret_then_rank(multi_bars(4))
    ranks = column_at(out, "ret_1_xrank", 2)
    assert ranks == {"A": pytest.approx(1 / 3), "B": pytest.approx(2 / 3), "C": pytest.approx(1.0)}


def test_descending_reverses() -> None:
    out = ret_then_rank(multi_bars(4), descending=True)
    ranks = column_at(out, "ret_1_xrank", 2)
    assert ranks["C"] == pytest.approx(1 / 3)


def test_raw_rank() -> None:
    out = ret_then_rank(multi_bars(4), pct=False)
    assert column_at(out, "ret_1_xrank", 2) == {"A": 1.0, "B": 2.0, "C": 3.0}


def test_warm_up_rows_are_not_members() -> None:
    """At bar 0 every ret_1 is null: no members, so every rank is null."""
    out = ret_then_rank(multi_bars(4))
    assert set(column_at(out, "ret_1_xrank", 0).values()) == {None}


def test_output_columns_and_order() -> None:
    out = ret_then_rank(multi_bars(3))
    assert out.columns == ["instrument_id", "bar_start", "ret_1", "ret_1_xrank", "available_at"]
    assert out.select("instrument_id", "bar_start").is_duplicated().sum() == 0


# -- missing-member policy ---------------------------------------------------------------


def test_a_gap_shrinks_the_cross_section() -> None:
    """B has no bar at 5: A and C are ranked among two, and B has no row at all."""
    out = ret_then_rank(multi_bars(8, skip={"B": {5}}))
    ranks = column_at(out, "ret_1_xrank", 5)
    assert ranks == {"A": pytest.approx(0.5), "C": pytest.approx(1.0)}
    # B's own next bar has a nulled ret_1 (its window spans the gap), so at 6 it is
    # present as a row but not a member.
    assert column_at(out, "ret_1", 6)["B"] is None
    assert column_at(out, "ret_1_xrank", 6) == {
        "A": pytest.approx(0.5),
        "B": None,
        "C": pytest.approx(1.0),
    }


def test_min_members_nulls_thin_cross_sections() -> None:
    out = ret_then_rank(multi_bars(8, skip={"B": {5}}), min_members=3)
    assert set(column_at(out, "ret_1_xrank", 5).values()) == {None}
    assert None not in column_at(out, "ret_1_xrank", 4).values()


def test_a_single_instrument_never_ranks() -> None:
    out = ret_then_rank(minute_bars(5))
    assert out.get_column("ret_1_xrank").null_count() == out.height


def test_min_members_below_two_is_refused() -> None:
    with pytest.raises(ValueError, match="min_members must be >= 2"):
        CrossSectionalRank("x", min_members=1)


# -- availability ----------------------------------------------------------------------


def test_availability_is_lifted_to_the_slowest_member() -> None:
    """B's bar 5 is 9 minutes late. A's and C's ranks at 5 cannot be known until B's
    value is, so the whole row -- ret_1 included -- waits."""
    bars = multi_bars(9, late={"B": {5: dt.timedelta(minutes=9)}})
    before = compute_features(bars, [LaggedReturn(1)])
    after = compute_cross_sectional(before, [CrossSectionalRank("ret_1")])
    late = at(6) + dt.timedelta(minutes=9)

    assert column_at(before, "available_at", 5)["A"] == at(6, 2), "per-instrument: A is on time"
    assert column_at(after, "available_at", 5) == {"A": late, "B": late, "C": late}
    # B's ret_1 at 6 also reads bar 5, so the batch at 6 waits too; at 7 it is free.
    assert set(column_at(after, "available_at", 6).values()) == {late}
    assert set(column_at(after, "available_at", 7).values()) == {at(8, 2)}


def test_availability_never_decreases() -> None:
    bars = multi_bars(9, late={"A": {3: dt.timedelta(minutes=4)}})
    before = compute_features(bars, [LaggedReturn(1)])
    after = compute_cross_sectional(before, [CrossSectionalRank("ret_1")])
    joined = before.join(after, on=["instrument_id", "bar_start"], suffix="_x")
    assert (joined.get_column("available_at_x") >= joined.get_column("available_at")).all()


# -- leakage ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bars",
    [
        multi_bars(40),
        multi_bars(40, skip={"B": {12, 13}}),
        multi_bars(40, late={"B": {9: dt.timedelta(minutes=6)}}),
        multi_bars(
            40,
            skip={"C": {20}},
            late={"A": {9: dt.timedelta(minutes=6)}, "B": {30: dt.timedelta(minutes=3)}},
        ),
    ],
    ids=["clean", "gap", "late", "gap_and_late"],
)
def test_ranks_pass_both_leakage_checks(bars: pl.DataFrame) -> None:
    assert_prefix_invariant(ret_then_rank, bars)
    assert_future_insensitive(ret_then_rank, bars)


def test_ranks_with_per_instrument_availability_would_leak() -> None:
    """The lift to batch availability is what makes ranks safe. Without it a late member
    changes an already-'available' rank -- the asynchrony leak, demonstrated."""

    def leaky(bars: pl.DataFrame) -> pl.DataFrame:
        frame = compute_features(bars, [LaggedReturn(1)])
        return frame.with_columns(
            CrossSectionalRank("ret_1").expression().alias("ret_1_xrank")
        )  # availability left per-instrument

    bars = multi_bars(40, late={"B": {9: dt.timedelta(minutes=6)}})
    from qresearch.features.leakage import LeakageDetected

    with pytest.raises(LeakageDetected):
        assert_prefix_invariant(leaky, bars)


# -- validation ------------------------------------------------------------------------


def test_missing_input_column_is_refused() -> None:
    frame = compute_features(multi_bars(3), [LaggedReturn(1)])
    with pytest.raises(ValueError, match="ranked columns \\['nope'\\]"):
        compute_cross_sectional(frame, [CrossSectionalRank("nope")])


def test_name_collision_is_refused() -> None:
    frame = compute_features(multi_bars(3), [LaggedReturn(1)]).with_columns(
        pl.lit(0.0).alias("ret_1_xrank")
    )
    with pytest.raises(ValueError, match="collide"):
        compute_cross_sectional(frame, [CrossSectionalRank("ret_1")])


def test_duplicate_rows_are_refused() -> None:
    frame = compute_features(multi_bars(3), [LaggedReturn(1)])
    with pytest.raises(ValueError, match="duplicate"):
        compute_cross_sectional(pl.concat([frame, frame]), [CrossSectionalRank("ret_1")])


def test_empty_rank_list_is_refused() -> None:
    with pytest.raises(ValueError, match="no cross-sectional"):
        compute_cross_sectional(compute_features(multi_bars(3), [LaggedReturn(1)]), [])
