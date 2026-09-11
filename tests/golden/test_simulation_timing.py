"""Golden timing scenarios, in the order DEVELOPMENT_PLAN.md sec. 5 prescribes.

Every number here is computed by hand in the comments. Prices are chosen so that each
bar's open, close, and the neighbouring bars' prints are all distinct, which makes a
wrong fill price identify *which* wrong print was used.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from pydantic import ValidationError
from tests.golden.conftest import Scripted, at, bars, free_config, run

from qresearch.simulation.events import OrderStatus
from qresearch.simulation.execution import ExecutionConfig, FillRule

# minute: open, close, volume. Every print distinct.
FOUR = [
    (0, 100.0, 101.0, 1000.0),
    (1, 102.0, 103.0, 1000.0),
    (2, 104.0, 105.0, 1000.0),
    (3, 106.0, 107.0, 1000.0),
]


# -- 1. one asset, one market order, no costs ------------------------------------------------


def test_one_order_fills_at_the_first_open_after_the_decision_could_have_been_acted_on() -> None:
    """Bar 0 publishes at 00:01:02. Bar 1 opened at 00:01:00 -- already printed. The
    first open the order can reach is bar 2's, at 00:02:00, price 104."""
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}))

    assert len(result.orders) == 1 and len(result.fills) == 1
    order, fill = result.orders[0], result.fills[0]
    assert order.signal_at == at(1, 2)
    assert order.order_at == at(1, 2) and order.eligible_at == at(1, 2)
    assert fill.fill_at == at(2, 0)
    assert fill.price == 104.0, "bar 2's open"
    assert (
        fill.reference_price == 104.0
        and fill.half_spread == 0
        and fill.slippage == 0
        and fill.fee == 0
    )
    assert fill.notional == 1040.0


def test_the_fill_price_is_not_any_print_the_strategy_had_seen() -> None:
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}))
    fill = result.fills[0]
    assert fill.price not in (100.0, 101.0), "bar 0 open/close: the signal bar"
    assert fill.price not in (102.0, 103.0), "bar 1 open/close: already printed / not yet known"


def test_cash_position_and_equity_after_the_fill() -> None:
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}))
    curve = {s.at: s for s in result.snapshots}
    # After the fill at 00:02:00: cash 100000 - 1040. Mark is still bar 1's close (103,
    # published 00:02:02 -- not yet!). Bar 0's close 101 is the latest published mark.
    after_fill = curve[at(2, 0)]
    assert after_fill.cash == pytest.approx(98_960.0)
    assert after_fill.equity == pytest.approx(98_960.0 + 10 * 101.0)
    # Bar 2 publishes at 00:03:02 with close 105: equity 98960 + 1050 = 100010.
    at_close = curve[at(3, 2)]
    assert at_close.equity == pytest.approx(100_010.0)
    assert at_close.unrealized_pnl == pytest.approx(10.0)
    assert at_close.realized_pnl == 0.0 and at_close.fees_paid == 0.0
    assert result.final.equity == pytest.approx(100_030.0), "bar 3 close 107"


def test_the_strategy_never_sees_an_unpublished_bar() -> None:
    strategy = Scripted({})
    run(bars("X", FOUR), strategy)
    for context in strategy.seen:
        assert context.bars_latest.get_column("available_at").max() <= context.decision_at
        assert context.bars_recent.get_column("available_at").max() <= context.decision_at
    assert [c.decision_at for c in strategy.seen] == [at(m, 2) for m in (1, 2, 3, 4)]


# -- 2. latency assertions ---------------------------------------------------------------------


def test_every_fill_satisfies_the_timestamp_chain_with_positive_latencies() -> None:
    config = free_config(
        execution=ExecutionConfig(
            submission_latency=dt.timedelta(milliseconds=500),
            order_latency=dt.timedelta(seconds=1),
            fill_rule=FillRule.NEXT_OPEN_AFTER_ELIGIBILITY,
            expire_after=None,
            participation_cap=None,
            liquidity_lookback_bars=1,
            costs=free_config().execution.costs,
        )
    )
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}), config=config)
    order, fill = result.orders[0], result.fills[0]
    assert order.signal_at == at(1, 2)
    assert order.order_at == at(1, 2.5)
    assert order.eligible_at == at(1, 3.5)
    assert order.signal_at < order.order_at < order.eligible_at < fill.fill_at
    assert fill.fill_at == at(2, 0)


def test_zero_publication_latency_still_cannot_reach_the_same_instants_open() -> None:
    """With bar 0 available at exactly 00:01:00 -- the instant bar 1 opens -- the open
    event is processed before the decision. The order fills at bar 2's open."""
    frame = bars("X", FOUR, latency=dt.timedelta(0))
    result = run(frame, Scripted({1: [("delta", "X", 10.0)]}))
    assert result.orders[0].signal_at == at(1, 0)
    assert result.fills[0].fill_at == at(2, 0)
    assert result.fills[0].price == 104.0


def test_a_long_order_latency_pushes_the_fill_to_a_later_open() -> None:
    config = free_config(
        execution=ExecutionConfig(
            order_latency=dt.timedelta(seconds=59),
            fill_rule=FillRule.NEXT_OPEN_AFTER_ELIGIBILITY,
            expire_after=None,
            participation_cap=None,
            liquidity_lookback_bars=1,
            costs=free_config().execution.costs,
        )
    )
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}), config=config)
    assert result.orders[0].eligible_at == at(2, 1)
    assert result.fills[0].fill_at == at(3, 0) and result.fills[0].price == 106.0


def test_the_optimistic_rule_fills_at_the_open_of_the_bar_containing_eligibility() -> None:
    """Eligible at 00:01:02 -> the bar containing it is bar 1 -> its open, 102. The fill
    is stamped at eligibility time and the run carries a warning."""
    config = free_config(
        execution=ExecutionConfig(
            order_latency=dt.timedelta(0),
            expire_after=None,
            participation_cap=None,
            liquidity_lookback_bars=1,
            fill_rule=FillRule.OPEN_OF_CURRENT_BAR,
            costs=free_config().execution.costs,
        )
    )
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}), config=config)
    fill = result.fills[0]
    assert fill.price == 102.0
    assert fill.fill_at == at(1, 2)
    assert "optimistic_fill_rule" in {w.code for w in result.warnings}


def test_orders_are_immutable_and_lifecycle_is_events() -> None:
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}))
    statuses = [(e.status, e.quantity) for e in result.order_events]
    assert statuses == [(OrderStatus.SUBMITTED, 10.0), (OrderStatus.FILLED, 10.0)]
    with pytest.raises(ValidationError):
        result.orders[0].quantity = 5.0  # type: ignore[misc]


def test_frames_have_typed_columns_even_when_empty() -> None:
    result = run(bars("X", FOUR), Scripted({}))
    frames = result.frames()
    assert frames["fills"].height == 0
    assert frames["fills"].schema["fill_at"] == pl.Datetime("us", "UTC")
    assert frames["equity_curve"].height > 0


def test_optional_ledger_columns_keep_their_numeric_and_temporal_types() -> None:
    """A persisted ``float | None`` must stay a float, not become a string that happens
    to round-trip through Parquet."""
    result = run(bars("X", FOUR), Scripted({1: [("delta", "X", 10.0)]}))
    frames = result.frames()
    assert frames["fills"].schema["participation"] == pl.Float64
    assert frames["orders"].schema["liquidity_estimate"] == pl.Float64
    assert frames["orders"].schema["expires_at"] == pl.Datetime("us", "UTC")
    assert frames["orders"].schema["order_id"] == pl.String
    assert frames["intent_outcomes"].schema["rejection"] == pl.String
    assert frames["warnings"].schema["occurrences"] == pl.Int64
