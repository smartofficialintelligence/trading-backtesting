"""Cost decomposition: every friction component is explicit, directional, and exact."""

from __future__ import annotations

import datetime as dt

import pytest
from tests.golden.conftest import Scripted, at, bars, free_config, run
from tests.golden.test_simulation_timing import FOUR

from qresearch.ids import InstrumentId, OrderId
from qresearch.simulation.events import IntentKind, Order, OrderSide
from qresearch.simulation.execution import (
    BarOpenExecutionModel,
    CostConfig,
    ExecutionConfig,
    FillRule,
    LiveOrder,
    OpenEvent,
    SlippageConfig,
    SlippageKind,
)

COSTLY = CostConfig(
    half_spread_bps=2.5,
    commission_bps=1.0,
    slippage=SlippageConfig(kind=SlippageKind.FIXED_BPS, coefficient=5.0),
)


def costly_config() -> object:
    return free_config(
        execution=ExecutionConfig(
            order_latency=dt.timedelta(0),
            expire_after=None,
            participation_cap=None,
            liquidity_lookback_bars=1,
            fill_rule=FillRule.NEXT_OPEN_AFTER_ELIGIBILITY,
            costs=COSTLY,
        )
    )


def test_buy_decomposition_by_hand() -> None:
    """Buy 10 at open 104: half-spread 0.026, slippage 0.052, price 104.078,
    fee 10 * 104.078 * 1e-4 = 0.104078."""
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}), config=costly_config())  # type: ignore[arg-type]
    fill = result.fills[0]
    assert fill.reference_price == 104.0
    assert fill.half_spread == pytest.approx(0.026)
    assert fill.slippage == pytest.approx(0.052)
    assert fill.price == pytest.approx(104.078)
    assert fill.fee == pytest.approx(0.104078)
    assert fill.total_cost == pytest.approx(0.26 + 0.52 + 0.104078)
    after = next(s for s in result.snapshots if s.at == at(2, 0))
    assert after.cash == pytest.approx(100_000 - 1040.78 - 0.104078)
    assert after.fees_paid == pytest.approx(0.104078)


def test_sell_pays_the_same_frictions_in_the_other_direction() -> None:
    """Buy 10 at bar 2's open, sell 10 at bar 3's open (106): price 106 - 0.0265 - 0.053."""
    result = run(
        bars("X", FOUR),
        Scripted({1: [("delta", "X", 10.0)], 2: [("delta", "X", -10.0)]}),
        config=costly_config(),  # type: ignore[arg-type]
    )
    sell = result.fills[1]
    assert sell.side is OrderSide.SELL and sell.reference_price == 106.0
    assert sell.price == pytest.approx(106.0 - 0.0265 - 0.053)
    assert sell.price < sell.reference_price, "sells receive less than the print"
    buy = result.fills[0]
    assert buy.price > buy.reference_price, "buys pay more than the print"


def test_costly_equity_is_free_equity_minus_exact_total_cost() -> None:
    free = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}))
    costly = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}), config=costly_config())  # type: ignore[arg-type]
    assert costly.final.equity < free.final.equity
    assert free.final.equity - costly.final.equity == pytest.approx(costly.total_costs)


def test_min_commission_and_per_unit_fee() -> None:
    costs = CostConfig(
        half_spread_bps=0,
        commission_bps=0,
        fee_per_unit=0.005,
        min_commission=1.0,
        slippage=SlippageConfig(kind=SlippageKind.FIXED_BPS, coefficient=0),
    )
    model = BarOpenExecutionModel(
        ExecutionConfig(costs=costs, participation_cap=None), quantity_increment=1.0
    )
    small = _order(10.0)
    result, _ = model.match(_event(104.0), [LiveOrder(order=small, remaining=10.0)], next_fill_id=1)
    assert result.fills[0].fee == 1.0, "10 * 0.005 = 0.05 < minimum 1.0"
    big = _order(1000.0)
    result, _ = model.match(_event(104.0), [LiveOrder(order=big, remaining=1000.0)], next_fill_id=1)
    assert result.fills[0].fee == pytest.approx(5.0)


def test_participation_slippage_is_linear_in_participation() -> None:
    costs = CostConfig(
        half_spread_bps=0,
        commission_bps=0,
        slippage=SlippageConfig(kind=SlippageKind.PARTICIPATION, coefficient=25.0),
    )
    model = BarOpenExecutionModel(
        ExecutionConfig(costs=costs, participation_cap=None), quantity_increment=1.0
    )
    order = _order(10.0, liquidity=1000.0)
    result, _ = model.match(_event(104.0), [LiveOrder(order=order, remaining=10.0)], next_fill_id=1)
    fill = result.fills[0]
    assert fill.participation == pytest.approx(0.01)
    assert fill.slippage == pytest.approx(104.0 * 25e-4 * 0.01)


def test_sqrt_impact_uses_volatility_and_participation() -> None:
    costs = CostConfig(
        half_spread_bps=0,
        commission_bps=0,
        slippage=SlippageConfig(kind=SlippageKind.SQRT_IMPACT, coefficient=1.0),
    )
    model = BarOpenExecutionModel(
        ExecutionConfig(costs=costs, participation_cap=None), quantity_increment=1.0
    )
    order = _order(10.0, liquidity=1000.0, volatility=0.01)
    result, _ = model.match(_event(104.0), [LiveOrder(order=order, remaining=10.0)], next_fill_id=1)
    assert result.fills[0].slippage == pytest.approx(104.0 * 1.0 * 0.01 * (10 / 1000) ** 0.5)


def test_missing_estimates_fall_back_with_a_warning() -> None:
    costs = CostConfig(
        half_spread_bps=0,
        commission_bps=0,
        slippage=SlippageConfig(
            kind=SlippageKind.PARTICIPATION, coefficient=25.0, fallback_bps=10.0
        ),
    )
    model = BarOpenExecutionModel(
        ExecutionConfig(costs=costs, participation_cap=None), quantity_increment=1.0
    )
    order = _order(10.0)  # no liquidity estimate
    result, _ = model.match(_event(104.0), [LiveOrder(order=order, remaining=10.0)], next_fill_id=1)
    assert result.fills[0].slippage == pytest.approx(104.0 * 10e-4)
    assert "slippage_fallback" in {code for code, _ in result.notes}


def test_default_costs_are_not_zero() -> None:
    """DEVELOPMENT_PLAN.md sec. 6: defaults that improve headline performance are dangerous."""
    defaults = CostConfig()
    assert defaults.half_spread_bps > 0 and defaults.commission_bps > 0
    assert defaults.slippage.coefficient > 0


def _event(open_: float) -> OpenEvent:
    return OpenEvent(instrument_id=InstrumentId("X"), at=at(2), open=open_, bar_start=at(2))


def _order(
    quantity: float, *, liquidity: float | None = None, volatility: float | None = None
) -> Order:
    return Order(
        order_id=OrderId("o1"),
        instrument_id=InstrumentId("X"),
        side=OrderSide.BUY,
        quantity=quantity,
        signal_at=at(1, 2),
        order_at=at(1, 2),
        eligible_at=at(1, 2),
        liquidity_estimate=liquidity,
        volatility_estimate=volatility,
        intent_kind=IntentKind.DELTA,
    )
