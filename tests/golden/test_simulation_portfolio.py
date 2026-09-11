"""Portfolio accounting, hand-checked through a long/short round trip."""

from __future__ import annotations

import pytest
from tests.golden.conftest import at

from qresearch.ids import InstrumentId, OrderId
from qresearch.simulation.events import Fill, OrderSide
from qresearch.simulation.portfolio import AccountingError, Portfolio

X = InstrumentId("X")


def fill(side: OrderSide, quantity: float, price: float, *, fee: float = 0.0, n: int = 1) -> Fill:
    return Fill(
        fill_id=f"f{n}",
        order_id=OrderId(f"o{n}"),
        instrument_id=X,
        side=side,
        quantity=quantity,
        fill_at=at(n),
        reference_price=price,
        half_spread=0.0,
        slippage=0.0,
        price=price,
        fee=fee,
        execution_model="test",
    )


def test_long_reduce_flip_and_close_by_hand() -> None:
    p = Portfolio(initial_cash=100_000.0)

    p.apply_fill(fill(OrderSide.BUY, 10, 104.0, n=1))  # cash 98960, long 10 @ 104
    assert p.cash == pytest.approx(98_960.0)
    assert p.positions[X].quantity == 10 and p.positions[X].avg_cost == 104.0

    p.apply_fill(fill(OrderSide.SELL, 4, 108.0, n=2))  # +432 cash; realized (108-104)*4 = 16
    assert p.cash == pytest.approx(99_392.0)
    assert p.realized_pnl == pytest.approx(16.0)
    assert p.positions[X].quantity == 6 and p.positions[X].avg_cost == 104.0

    p.apply_fill(fill(OrderSide.SELL, 10, 110.0, n=3))  # close 6: +36 -> 52; short 4 @ 110
    assert p.cash == pytest.approx(100_492.0)
    assert p.realized_pnl == pytest.approx(52.0)
    assert p.positions[X].quantity == -4 and p.positions[X].avg_cost == 110.0

    p.set_mark(X, 112.0, at(4))
    assert p.unrealized_pnl == pytest.approx(-8.0), "short 4, mark up 2"
    assert p.equity == pytest.approx(100_044.0)
    p.reconcile()

    p.apply_fill(fill(OrderSide.BUY, 4, 111.0, n=5))  # cover: realized (111-110)*(-4) = -4 -> 48
    assert p.positions[X].quantity == 0 and p.positions[X].avg_cost == 0.0
    assert p.realized_pnl == pytest.approx(48.0)
    assert p.cash == pytest.approx(100_048.0)
    assert p.equity == pytest.approx(100_048.0)
    p.reconcile()


def test_fees_reduce_cash_and_equity_but_not_pnl() -> None:
    p = Portfolio(initial_cash=100_000.0)
    p.apply_fill(fill(OrderSide.BUY, 10, 104.0, fee=1.0))
    p.set_mark(X, 104.0, at(2))
    assert p.cash == pytest.approx(98_959.0)
    assert p.fees_paid == 1.0
    assert p.realized_pnl == 0.0 and p.unrealized_pnl == 0.0
    assert p.equity == pytest.approx(99_999.0)
    p.reconcile()


def test_adding_to_a_position_blends_the_cost_basis() -> None:
    p = Portfolio(initial_cash=100_000.0)
    p.apply_fill(fill(OrderSide.BUY, 10, 100.0, n=1))
    p.apply_fill(fill(OrderSide.BUY, 30, 120.0, n=2))
    assert p.positions[X].avg_cost == pytest.approx(115.0)
    p.set_mark(X, 115.0, at(3))
    assert p.unrealized_pnl == pytest.approx(0.0)
    p.reconcile()


def test_exposures() -> None:
    p = Portfolio(initial_cash=100_000.0)
    y = InstrumentId("Y")
    p.apply_fill(fill(OrderSide.BUY, 10, 100.0, n=1))
    p.apply_fill(fill(OrderSide.SELL, 5, 200.0, n=2).model_copy(update={"instrument_id": y}))
    p.set_mark(X, 100.0, at(3))
    p.set_mark(y, 200.0, at(3))
    assert p.gross_exposure == pytest.approx(2000.0)
    assert p.net_exposure == pytest.approx(0.0)


def test_valuing_a_position_without_a_mark_is_an_error() -> None:
    p = Portfolio(initial_cash=100_000.0)
    p.apply_fill(fill(OrderSide.BUY, 10, 104.0))
    with pytest.raises(AccountingError, match="no mark"):
        _ = p.equity


def test_reconcile_detects_a_corrupted_ledger() -> None:
    p = Portfolio(initial_cash=100_000.0)
    p.apply_fill(fill(OrderSide.BUY, 10, 104.0))
    p.set_mark(X, 104.0, at(2))
    p.cash += 1.0  # tamper
    with pytest.raises(AccountingError):
        p.reconcile()


def test_snapshot_counts_open_positions_only() -> None:
    p = Portfolio(initial_cash=100_000.0)
    p.apply_fill(fill(OrderSide.BUY, 10, 104.0, n=1))
    p.apply_fill(fill(OrderSide.SELL, 10, 104.0, n=2))
    p.set_mark(X, 104.0, at(3))
    assert p.snapshot(at(3)).position_count == 0
    assert p.position_snapshots(at(3)) == []
