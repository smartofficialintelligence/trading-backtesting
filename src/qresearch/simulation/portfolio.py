"""Portfolio accounting: an immutable fill ledger and the state derived from it.

Accounting is a chronological state machine, not a vectorised calculation
(ARCHITECTURE.md sec. 2), because vectorisation hides timing errors. It is independent
of strategy and metrics.

Numerical policy: prices, quantities, and cash are IEEE-754 doubles. The accounting
identity checked by :meth:`Portfolio.reconcile` is::

    equity == initial_cash + realized_pnl + unrealized_pnl - fees_paid

with a stated tolerance. Cost basis uses signed quantities so one update rule covers long
and short positions.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from qresearch.ids import InstrumentId
from qresearch.simulation.events import Fill, PortfolioSnapshot, PositionSnapshot

RECONCILE_RTOL = 1e-9


class AccountingError(AssertionError):
    """The ledger no longer satisfies its own identity."""


@dataclass(slots=True)
class Position:
    quantity: float = 0.0
    """Signed units: negative is short."""

    avg_cost: float = 0.0
    """Average entry price of the open quantity. Zero when flat."""

    realized_pnl: float = 0.0

    def apply(self, signed_quantity: float, price: float) -> None:
        q0 = self.quantity
        new_q = q0 + signed_quantity
        if q0 == 0.0 or (q0 > 0) == (signed_quantity > 0):
            # Opening or adding: blend the cost basis.
            self.avg_cost = (q0 * self.avg_cost + signed_quantity * price) / new_q
            self.quantity = new_q
            return
        # Reducing, closing, or flipping. The closed portion realises against avg_cost.
        closed = signed_quantity if abs(signed_quantity) <= abs(q0) else -q0
        self.realized_pnl += (price - self.avg_cost) * (-closed)
        self.quantity = new_q
        if new_q == 0.0:
            self.avg_cost = 0.0
        elif (new_q > 0) != (q0 > 0):
            self.avg_cost = price  # flipped: what remains was opened at this fill

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0


@dataclass(slots=True)
class Mark:
    price: float
    at: _dt.datetime


@dataclass(slots=True)
class Portfolio:
    initial_cash: float
    cash: float = field(init=False)
    fees_paid: float = field(init=False, default=0.0)
    positions: dict[InstrumentId, Position] = field(init=False, default_factory=dict)
    marks: dict[InstrumentId, Mark] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")
        self.cash = self.initial_cash

    # -- events --------------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        signed = fill.side.sign * fill.quantity
        self.cash -= signed * fill.price
        self.cash -= fill.fee
        self.fees_paid += fill.fee
        self.positions.setdefault(fill.instrument_id, Position()).apply(signed, fill.price)

    def set_mark(self, instrument_id: InstrumentId, price: float, at: _dt.datetime) -> None:
        if price <= 0:
            raise ValueError(f"mark for {instrument_id} must be positive, got {price}")
        self.marks[instrument_id] = Mark(price=price, at=at)

    # -- derived state ---------------------------------------------------------------

    def quantity(self, instrument_id: InstrumentId) -> float:
        position = self.positions.get(instrument_id)
        return 0.0 if position is None else position.quantity

    def market_value(self, instrument_id: InstrumentId) -> float:
        position = self.positions.get(instrument_id)
        if position is None or position.is_flat:
            return 0.0
        return position.quantity * self._mark_for(instrument_id).price

    def _mark_for(self, instrument_id: InstrumentId) -> Mark:
        try:
            return self.marks[instrument_id]
        except KeyError:
            raise AccountingError(
                f"no mark for {instrument_id}, which has an open position; a portfolio "
                "cannot be valued without a point-in-time price for every holding"
            ) from None

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        total = 0.0
        for instrument_id, position in self.positions.items():
            if not position.is_flat:
                total += (
                    self._mark_for(instrument_id).price - position.avg_cost
                ) * position.quantity
        return total

    @property
    def equity(self) -> float:
        return self.cash + sum(self.market_value(i) for i in self.positions)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(self.market_value(i)) for i in self.positions)

    @property
    def net_exposure(self) -> float:
        return sum(self.market_value(i) for i in self.positions)

    def reconcile(self, *, rtol: float = RECONCILE_RTOL) -> None:
        """Assert the accounting identity. Raises :class:`AccountingError`."""
        lhs = self.equity
        rhs = self.initial_cash + self.realized_pnl + self.unrealized_pnl - self.fees_paid
        scale = max(abs(lhs), abs(rhs), self.initial_cash, 1.0)
        if abs(lhs - rhs) > rtol * scale:
            raise AccountingError(
                f"equity {lhs!r} != initial {self.initial_cash!r} + realized "
                f"{self.realized_pnl!r} + unrealized {self.unrealized_pnl!r} - fees "
                f"{self.fees_paid!r} = {rhs!r}"
            )

    def snapshot(self, at: _dt.datetime, *, pending_orders: int = 0) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            at=at,
            cash=self.cash,
            equity=self.equity,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            fees_paid=self.fees_paid,
            position_count=sum(1 for p in self.positions.values() if not p.is_flat),
            pending_orders=pending_orders,
        )

    def position_snapshots(self, at: _dt.datetime) -> list[PositionSnapshot]:
        out = []
        for instrument_id in sorted(self.positions):
            position = self.positions[instrument_id]
            if position.is_flat:
                continue
            mark = self._mark_for(instrument_id)
            out.append(
                PositionSnapshot(
                    at=at,
                    instrument_id=instrument_id,
                    quantity=position.quantity,
                    avg_cost=position.avg_cost,
                    mark=mark.price,
                    mark_at=mark.at,
                    market_value=position.quantity * mark.price,
                    unrealized_pnl=(mark.price - position.avg_cost) * position.quantity,
                )
            )
        return out
