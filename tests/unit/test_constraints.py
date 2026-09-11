"""Intent sizing and constraints: exact numbers, and order independence."""

from __future__ import annotations

from tests.golden.conftest import at

from qresearch.ids import InstrumentId
from qresearch.simulation.constraints import (
    ConstraintConfig,
    SizingState,
    floor_to_increment,
    size_intents,
)
from qresearch.simulation.events import OrderIntent, RejectionReason

T = at(5)
X, Y, Z = InstrumentId("X"), InstrumentId("Y"), InstrumentId("Z")


def state(**overrides: object) -> SizingState:
    base: dict[str, object] = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "marks": {X: 100.0, Y: 50.0, Z: 10.0},
        "positions": {},
        "pending": {},
        "increments": {X: 1.0, Y: 1.0, Z: 1.0},
    }
    return SizingState(**(base | overrides))  # type: ignore[arg-type]


def size(intents: list[OrderIntent], s: SizingState | None = None, **cfg: object) -> list[float]:
    outcomes = size_intents(intents, s or state(), ConstraintConfig.model_validate(cfg), at=T)
    return [o.final_quantity for o in outcomes]


def test_target_weight_uses_equity_and_mark() -> None:
    assert size([OrderIntent.target_weight("X", 0.5, at=T)]) == [500.0]


def test_targets_net_against_position_and_pending() -> None:
    s = state(positions={X: 100.0}, pending={X: 50.0})
    assert size([OrderIntent.target_quantity("X", 175.0, at=T)], s) == [25.0]
    assert size([OrderIntent.target_quantity("X", 150.0, at=T)], s) == [0.0], "already on its way"


def test_rounding_toward_zero_to_the_increment() -> None:
    assert floor_to_increment(55.0, 10.0) == 50.0
    assert floor_to_increment(-55.0, 10.0) == -50.0
    assert floor_to_increment(0.3, 1.0) == 0.0
    s = state(increments={X: 10.0, Y: 1.0, Z: 1.0})
    out = size_intents([OrderIntent.delta("X", 55.0, at=T)], s, ConstraintConfig(), at=T)[0]
    assert out.final_quantity == 50.0 and "rounded" in (out.note or "")


def test_below_increment_is_rejected() -> None:
    s = state(increments={X: 10.0, Y: 1.0, Z: 1.0})
    out = size_intents([OrderIntent.delta("X", 5.0, at=T)], s, ConstraintConfig(), at=T)[0]
    assert out.final_quantity == 0.0 and out.rejection is RejectionReason.BELOW_INCREMENT


def test_short_not_allowed_resizes_to_flat_or_rejects() -> None:
    flat = size_intents([OrderIntent.delta("X", -10.0, at=T)], state(), ConstraintConfig(), at=T)[0]
    assert flat.rejection is RejectionReason.SHORT_NOT_ALLOWED
    partial = size_intents(
        [OrderIntent.delta("X", -10.0, at=T)], state(positions={X: 6.0}), ConstraintConfig(), at=T
    )[0]
    assert partial.final_quantity == -6.0 and "shorting not allowed" in (partial.note or "")
    assert size([OrderIntent.delta("X", -10.0, at=T)], allow_short=True) == [-10.0]


def test_position_limit_caps_a_single_name() -> None:
    out = size_intents(
        [OrderIntent.target_weight("X", 0.5, at=T)],
        state(),
        ConstraintConfig(max_position_weight=0.3),
        at=T,
    )[0]
    assert out.final_quantity == 300.0 and "position limit" in (out.note or "")


def test_position_limit_does_not_block_reducing_orders() -> None:
    s = state(positions={X: 900.0}, cash=10_000.0)
    out = size_intents(
        [OrderIntent.delta("X", -100.0, at=T)], s, ConstraintConfig(max_position_weight=0.3), at=T
    )[0]
    assert out.final_quantity == -100.0


def test_gross_limit_scales_all_increasing_orders_by_one_factor() -> None:
    """X 0.7 + Y 0.7 = 1.4x equity; limit 1.0 -> both scaled by 5/7."""
    intents = [OrderIntent.target_weight("X", 0.7, at=T), OrderIntent.target_weight("Y", 0.7, at=T)]
    config = ConstraintConfig(max_gross_exposure=1.0, cost_buffer_bps=0.0)
    out = size_intents(intents, state(), config, at=T)
    assert [o.final_quantity for o in out] == [500.0, 1000.0]
    assert all("gross exposure" in (o.note or "") for o in out)


def test_gross_limit_counts_untouched_holdings() -> None:
    s = state(positions={Z: 5_000.0}, cash=50_000.0)  # Z worth 50k
    config = ConstraintConfig(max_gross_exposure=1.0, cost_buffer_bps=0.0)
    out = size_intents([OrderIntent.target_weight("X", 0.8, at=T)], s, config, at=T)[0]
    assert out.final_quantity == 500.0, "only 50k of room left"


def test_cash_check_scales_buys_with_a_cost_buffer() -> None:
    s = state(cash=50_000.0, positions={Z: 5_000.0})
    out = size_intents(
        [OrderIntent.target_weight("X", 0.5, at=T)], s, ConstraintConfig(cost_buffer_bps=10.0), at=T
    )[0]
    assert out.final_quantity == 499.0, "50000 / (100 * 1.001) = 499.5 -> 499"
    assert "cash" in (out.note or "")


def test_sells_fund_buys_in_the_same_batch() -> None:
    s = state(cash=0.0, positions={Y: 2_000.0})  # Y worth 100k; no cash
    intents = [OrderIntent.target_weight("Y", 0.0, at=T), OrderIntent.target_weight("X", 0.9, at=T)]
    out = size_intents(intents, s, ConstraintConfig(cost_buffer_bps=0.0), at=T)
    assert [o.final_quantity for o in out] == [-2000.0, 900.0]


def test_intent_order_does_not_change_the_result() -> None:
    a = [OrderIntent.target_weight("X", 0.7, at=T), OrderIntent.target_weight("Y", 0.7, at=T)]
    b = list(reversed(a))
    out_a = {
        o.instrument_id: o.final_quantity
        for o in size_intents(a, state(), ConstraintConfig(), at=T)
    }
    out_b = {
        o.instrument_id: o.final_quantity
        for o in size_intents(b, state(), ConstraintConfig(), at=T)
    }
    assert out_a == out_b


def test_stale_unknown_and_unmarked_intents_are_rejected() -> None:
    stale = size_intents(
        [OrderIntent.delta("X", 1.0, at=at(4))], state(), ConstraintConfig(), at=T
    )[0]
    assert stale.rejection is RejectionReason.STALE_INTENT
    unknown = size_intents([OrderIntent.delta("Q", 1.0, at=T)], state(), ConstraintConfig(), at=T)[
        0
    ]
    assert unknown.rejection is RejectionReason.UNKNOWN_INSTRUMENT
    unmarked = size_intents(
        [OrderIntent.target_weight("X", 0.5, at=T)], state(marks={}), ConstraintConfig(), at=T
    )[0]
    assert unmarked.rejection is RejectionReason.NO_MARK
    assert size([OrderIntent.delta("X", 5.0, at=T)], state(marks={})) == [5.0], (
        "delta needs no mark"
    )


def test_the_cash_buffer_applies_after_the_gross_limit() -> None:
    """Defaults compose: gross limit to 500, then a 10 bps buffer makes 500 unaffordable."""
    intents = [OrderIntent.target_weight("X", 0.7, at=T), OrderIntent.target_weight("Y", 0.7, at=T)]
    out = size_intents(intents, state(), ConstraintConfig(max_gross_exposure=1.0), at=T)
    assert [o.final_quantity for o in out] == [499.0, 999.0]
