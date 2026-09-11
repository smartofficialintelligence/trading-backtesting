"""Order, fill, and ledger records.

Every timestamp in the order lifecycle is a separate field (ARCHITECTURE.md sec. 2):
``signal_at <= order_at <= eligible_at <= fill_at``. They are never collapsed, and the
engine asserts the chain on every fill.
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.ids import InstrumentId, OrderId
from qresearch.time import UtcDatetime


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1


class IntentKind(StrEnum):
    DELTA = "delta"
    """Buy/sell this many units (signed)."""

    TARGET_QUANTITY = "target_quantity"
    """Hold this many units after the order (signed)."""

    TARGET_WEIGHT = "target_weight"
    """Hold this fraction of equity (signed) after the order."""


class OrderIntent(FrozenModel):
    """What a strategy asks for. Converted to an :class:`Order` by the engine."""

    instrument_id: InstrumentId
    kind: IntentKind
    value: float
    """Units for DELTA/TARGET_QUANTITY, a fraction of equity for TARGET_WEIGHT."""

    created_at: UtcDatetime
    """The decision instant. Set by the strategy from the context; checked by the engine."""

    tag: str | None = None
    """Free-form strategy annotation carried onto the order for later analysis."""

    @classmethod
    def delta(
        cls, instrument_id: str, units: float, *, at: _dt.datetime, tag: str | None = None
    ) -> OrderIntent:
        return cls(
            instrument_id=InstrumentId(instrument_id),
            kind=IntentKind.DELTA,
            value=units,
            created_at=at,
            tag=tag,
        )

    @classmethod
    def target_quantity(
        cls, instrument_id: str, units: float, *, at: _dt.datetime, tag: str | None = None
    ) -> OrderIntent:
        return cls(
            instrument_id=InstrumentId(instrument_id),
            kind=IntentKind.TARGET_QUANTITY,
            value=units,
            created_at=at,
            tag=tag,
        )

    @classmethod
    def target_weight(
        cls, instrument_id: str, weight: float, *, at: _dt.datetime, tag: str | None = None
    ) -> OrderIntent:
        return cls(
            instrument_id=InstrumentId(instrument_id),
            kind=IntentKind.TARGET_WEIGHT,
            value=weight,
            created_at=at,
            tag=tag,
        )


class OrderStatus(StrEnum):
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class Order(FrozenModel):
    """A validated, sized instruction. Immutable; state changes are :class:`OrderEvent`s."""

    order_id: OrderId
    instrument_id: InstrumentId
    side: OrderSide
    quantity: float = Field(gt=0)
    signal_at: UtcDatetime
    order_at: UtcDatetime
    eligible_at: UtcDatetime
    expires_at: UtcDatetime | None = None
    liquidity_estimate: float | None = None
    """Trailing per-bar volume estimate, from data available by ``order_at``."""

    volatility_estimate: float | None = None
    """Trailing 1-bar return std, from data available by ``order_at``."""

    intent_kind: IntentKind
    tag: str | None = None

    @model_validator(mode="after")
    def _check_chain(self) -> Self:
        if not (self.signal_at <= self.order_at <= self.eligible_at):
            raise ValueError(
                f"order {self.order_id}: need signal_at <= order_at <= eligible_at, got "
                f"{self.signal_at} / {self.order_at} / {self.eligible_at}"
            )
        if self.expires_at is not None and self.expires_at < self.order_at:
            raise ValueError(f"order {self.order_id}: expires_at precedes order_at")
        return self


class OrderEvent(FrozenModel):
    order_id: OrderId
    at: UtcDatetime
    status: OrderStatus
    quantity: float = Field(ge=0)
    """Quantity involved in this event (filled amount, or remaining for expiry)."""

    reason: str | None = None


class Fill(FrozenModel):
    """One execution, with the price decomposed so cost assumptions are auditable.

    ``price = reference_price + side.sign * (half_spread + slippage)``; ``fee`` is a
    separate cash entry.
    """

    fill_id: str
    order_id: OrderId
    instrument_id: InstrumentId
    side: OrderSide
    quantity: float = Field(gt=0)
    fill_at: UtcDatetime
    reference_price: float = Field(gt=0)
    half_spread: float = Field(ge=0)
    """Per-unit spread cost applied in the adverse direction."""

    slippage: float = Field(ge=0)
    """Per-unit impact cost applied in the adverse direction."""

    price: float = Field(gt=0)
    fee: float = Field(ge=0)
    execution_model: str
    participation: float | None = None
    """``quantity / liquidity_estimate`` where an estimate existed."""

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def spread_cost(self) -> float:
        return self.quantity * self.half_spread

    @property
    def slippage_cost(self) -> float:
        return self.quantity * self.slippage

    @property
    def total_cost(self) -> float:
        """All friction versus the reference price: spread + slippage + fee."""
        return self.spread_cost + self.slippage_cost + self.fee


class RejectionReason(StrEnum):
    UNKNOWN_INSTRUMENT = "unknown_instrument"
    BELOW_INCREMENT = "below_increment"
    SHORT_NOT_ALLOWED = "short_not_allowed"
    GROSS_EXPOSURE = "gross_exposure"
    POSITION_LIMIT = "position_limit"
    INSUFFICIENT_CASH = "insufficient_cash"
    NO_MARK = "no_mark"
    STALE_INTENT = "stale_intent"


class IntentOutcome(FrozenModel):
    """What happened to one intent: sized into an order, resized, or rejected."""

    at: UtcDatetime
    instrument_id: InstrumentId
    intent_kind: IntentKind
    intent_value: float
    requested_quantity: float
    """Signed units the intent implied before constraints."""

    final_quantity: float
    """Signed units actually ordered (0 if rejected)."""

    order_id: OrderId | None = None
    rejection: RejectionReason | None = None
    note: str | None = None


class WarningRecord(FrozenModel):
    at: UtcDatetime
    code: str
    message: str
    instrument_id: InstrumentId | None = None
    occurrences: int = Field(default=1, ge=1)


class PositionSnapshot(FrozenModel):
    at: UtcDatetime
    instrument_id: InstrumentId
    quantity: float
    avg_cost: float
    mark: float
    mark_at: UtcDatetime
    market_value: float
    unrealized_pnl: float


class PortfolioSnapshot(FrozenModel):
    at: UtcDatetime
    cash: float
    equity: float
    gross_exposure: float
    net_exposure: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    position_count: int
    pending_orders: int
