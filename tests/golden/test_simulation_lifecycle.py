"""Order lifecycle: participation limits, expiry, gaps, liquidation, staleness, shorts."""

from __future__ import annotations

import datetime as dt

import pytest

from qresearch.simulation.constraints import ConstraintConfig
from qresearch.simulation.engine import EndOfRunPolicy
from qresearch.simulation.events import OrderStatus, RejectionReason
from qresearch.simulation.execution import CostConfig, ExecutionConfig
from tests.golden.conftest import Scripted, at, bars, free_config, run

SIX = [(m, 100.0 + 2 * m, 101.0 + 2 * m, 100.0) for m in range(6)]


def execution(**overrides: object) -> ExecutionConfig:
    base: dict[str, object] = {
        "order_latency": dt.timedelta(0),
        "expire_after": None,
        "participation_cap": None,
        "liquidity_lookback_bars": 1,
        "costs": CostConfig.free(),
    }
    return ExecutionConfig.model_validate(base | overrides)


# -- participation ---------------------------------------------------------------------------


def test_participation_cap_spreads_a_large_order_over_several_opens() -> None:
    """Trailing volume 100, cap 10% -> 10 units per open. 25 units fill 10, 10, 5."""
    config = free_config(execution=execution(participation_cap=0.1, liquidity_lookback_bars=2))
    result = run(bars("X", SIX), Scripted({2: [("delta", "X", 25.0)]}), config=config)
    assert [(f.fill_at, f.quantity, f.price) for f in result.fills] == [
        (at(3), 10.0, 106.0),
        (at(4), 10.0, 108.0),
        (at(5), 5.0, 110.0),
    ]
    statuses = [e.status for e in result.order_events]
    assert statuses == [
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
    ]
    assert result.orders[0].liquidity_estimate == 100.0
    assert "participation_limited" in {w.code for w in result.warnings}
    assert result.final.equity == pytest.approx(
        100_000 + 10 * (111 - 106) + 10 * (111 - 108) + 5 * (111 - 110)
    )


def test_the_liquidity_estimate_uses_only_bars_published_by_the_order() -> None:
    """Lookback 3 at the third decision: bars 0, 1, 2 (volumes 100) are published; the
    estimate must not include bar 3's volume even though it is in the frame."""
    frame = bars(
        "X",
        [
            (0, 100.0, 101.0, 100.0),
            (1, 102.0, 103.0, 100.0),
            (2, 104.0, 105.0, 100.0),
            (3, 106.0, 107.0, 9_999.0),
            (4, 108.0, 109.0, 9_999.0),
            (5, 110.0, 111.0, 100.0),
        ],
    )
    config = free_config(execution=execution(participation_cap=0.1, liquidity_lookback_bars=3))
    result = run(frame, Scripted({3: [("delta", "X", 5.0)]}), config=config)
    assert result.orders[0].liquidity_estimate == 100.0


def test_no_estimate_means_no_cap_but_a_warning() -> None:
    config = free_config(execution=execution(participation_cap=0.1, liquidity_lookback_bars=5))
    result = run(bars("X", SIX), Scripted({1: [("delta", "X", 25.0)]}), config=config)
    assert result.orders[0].liquidity_estimate is None
    assert result.fills[0].quantity == 25.0
    assert "no_liquidity_estimate" in {w.code for w in result.warnings}


# -- expiry and gaps ----------------------------------------------------------------------------


GAPPED = [
    (0, 100.0, 101.0, 100.0),
    (1, 102.0, 103.0, 100.0),
    (2, 104.0, 105.0, 100.0),
    (10, 120.0, 121.0, 100.0),
]


def test_an_order_waits_through_a_gap_and_fills_at_the_next_real_open() -> None:
    """No bars 3..9. The order placed at 00:03:02 fills at bar 10's open, never at a
    stale or interpolated price."""
    result = run(bars("X", GAPPED), Scripted({3: [("delta", "X", 10.0)]}))
    assert len(result.fills) == 1
    assert result.fills[0].fill_at == at(10) and result.fills[0].price == 120.0


def test_an_order_expires_before_a_late_open_arrives() -> None:
    config = free_config(execution=execution(expire_after=dt.timedelta(minutes=5)))
    result = run(bars("X", GAPPED), Scripted({3: [("delta", "X", 10.0)]}), config=config)
    assert result.fills == []
    expired = [e for e in result.order_events if e.status is OrderStatus.EXPIRED]
    assert len(expired) == 1 and expired[0].quantity == 10.0
    assert result.orders[0].expires_at == at(8, 2)
    assert expired[0].at == at(10), "processed at the next instant, before that open"
    assert result.final.position_count == 0


def test_expiry_at_exactly_the_open_instant_expires_first() -> None:
    """Conservative tie-break: an order whose expiry coincides with an open does not fill."""
    config = free_config(execution=execution(expire_after=at(10) - at(3, 2)))
    result = run(bars("X", GAPPED), Scripted({3: [("delta", "X", 10.0)]}), config=config)
    assert result.orders[0].expires_at == at(10)
    assert result.fills == []


# -- end of run ---------------------------------------------------------------------------------


def test_liquidation_flattens_at_the_first_open_after_the_range() -> None:
    config = free_config(end_of_run=EndOfRunPolicy.LIQUIDATE)
    result = run(bars("X", SIX), Scripted({1: [("delta", "X", 10.0)]}), config=config, end=3)
    liquidation = [o for o in result.orders if o.tag == "liquidation"]
    assert len(liquidation) == 1 and liquidation[0].signal_at == at(3)
    assert result.fills[-1].fill_at == at(4) and result.fills[-1].price == 108.0
    assert result.final.position_count == 0
    assert result.final.equity == pytest.approx(100_000 + 10 * (108 - 104))
    assert result.final.equity == pytest.approx(result.final.cash)


def test_an_order_that_can_fill_at_the_range_end_instant_does_so_before_liquidation() -> None:
    """Phase order: at 00:04:00 (both an open and the range end) fills precede the
    liquidation step, so an in-range order that can fill is honoured."""
    config = free_config(end_of_run=EndOfRunPolicy.LIQUIDATE)
    result = run(
        bars("X", SIX),
        Scripted({1: [("delta", "X", 10.0)], 3: [("delta", "X", 10.0)]}),
        config=config,
        end=4,
    )
    assert [f.fill_at for f in result.fills][:2] == [at(2), at(4)]
    assert not [e for e in result.order_events if e.status is OrderStatus.CANCELLED]
    assert result.final.position_count == 0


def test_liquidation_cancels_orders_that_have_no_open_to_fill_at() -> None:
    """Bars 0..3 then 10. The order placed at 00:03:02 has no open before the range end
    (00:04:00); the first instant past the end is bar 3's publication at 00:04:02, where
    liquidation cancels it and flattens at bar 10's open."""
    frame = bars(
        "X",
        [
            (0, 100.0, 101.0, 100.0),
            (1, 102.0, 103.0, 100.0),
            (2, 104.0, 105.0, 100.0),
            (3, 106.0, 107.0, 100.0),
            (10, 120.0, 121.0, 100.0),
        ],
    )
    config = free_config(end_of_run=EndOfRunPolicy.LIQUIDATE)
    result = run(
        frame,
        Scripted({1: [("delta", "X", 10.0)], 3: [("delta", "X", 10.0)]}),
        config=config,
        end=4,
    )
    cancelled = [e for e in result.order_events if e.status is OrderStatus.CANCELLED]
    assert len(cancelled) == 1 and cancelled[0].reason == "cancelled for liquidation"
    assert cancelled[0].at == at(4, 2)
    assert result.fills[-1].fill_at == at(10) and result.fills[-1].price == 120.0
    assert result.final.position_count == 0


def test_liquidation_impossible_is_a_warning_not_a_silent_mark() -> None:
    config = free_config(end_of_run=EndOfRunPolicy.LIQUIDATE)
    result = run(bars("X", SIX), Scripted({1: [("delta", "X", 10.0)]}), config=config, end=100)
    assert "liquidation_impossible" in {w.code for w in result.warnings}
    assert result.final.position_count == 1


def test_mark_policy_leaves_positions_open_and_valued_at_the_last_close() -> None:
    result = run(bars("X", SIX), Scripted({1: [("delta", "X", 10.0)]}))
    assert result.final.position_count == 1
    assert result.positions[-1].mark == 111.0


# -- staleness ---------------------------------------------------------------------------------


def test_a_position_valued_at_a_stale_mark_is_flagged() -> None:
    """X stops publishing after bar 2 while Y keeps going. Every decision after 00:08:02
    values the X position at a mark older than the 5-minute limit."""
    import polars as pl

    x = bars("X", [(0, 100.0, 101.0, 100.0), (1, 102.0, 103.0, 100.0), (2, 104.0, 105.0, 100.0)])
    y = bars("Y", [(m, 50.0, 51.0, 100.0) for m in range(12)])
    config = free_config(max_mark_staleness=dt.timedelta(minutes=5))
    from tests.golden.conftest import instrument

    result = run(
        pl.concat([x, y]),
        Scripted({1: [("delta", "X", 10.0)]}),
        config=config,
        instruments={"X": instrument("X"), "Y": instrument("Y")},
    )
    stale = [w for w in result.warnings if w.code == "stale_mark"]
    assert len(stale) == 1 and stale[0].instrument_id == "X"
    assert stale[0].at == at(9, 2), "mark from 00:03:02 is 6 minutes old at 00:09:02"
    assert stale[0].occurrences == 4, "and every decision after that, deduplicated"


# -- shorts -------------------------------------------------------------------------------------


def test_shorting_is_rejected_by_default() -> None:
    result = run(bars("X", SIX), Scripted({1: [("delta", "X", -10.0)]}))
    assert result.orders == []
    assert result.outcomes[0].rejection is RejectionReason.SHORT_NOT_ALLOWED


def test_shorting_when_allowed_carries_a_stated_limitation() -> None:
    config = free_config(constraints=ConstraintConfig(allow_short=True))
    result = run(bars("X", SIX), Scripted({1: [("delta", "X", -10.0)]}), config=config)
    assert result.fills[0].quantity == 10.0
    assert result.final.position_count == 1
    assert "short_position" in {w.code for w in result.warnings}
    assert result.final.equity == pytest.approx(100_000 - 10 * (111 - 104))
