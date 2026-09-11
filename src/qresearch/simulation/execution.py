"""Bar-open execution with explicit, decomposed costs.

The only execution model in the MVP (DEVELOPMENT_PLAN.md sec. 3): market orders filled at
a bar open. Everything the model may look at is stated:

* the **opening price** of the bar whose open event is being processed -- not that bar's
  high, low, close, or volume, which do not exist yet;
* the order's ``liquidity_estimate`` and ``volatility_estimate``, which the engine
  computed from bars available by ``order_at``.

Fill price = ``open + side * (half_spread + slippage)``. Fees are a separate entry. Each
component is recorded on the :class:`~qresearch.simulation.events.Fill` so a result can
be decomposed into "what the market did" and "what the assumptions cost".

Costs are non-zero by default (DEVELOPMENT_PLAN.md sec. 6: "defaults that improve
headline performance are dangerous").
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.ids import InstrumentId, OrderId
from qresearch.simulation.events import Fill, Order


class FillRule(StrEnum):
    OPEN_OF_CURRENT_BAR = "open_of_current_bar"
    """Fill at the open of the bar *containing* ``eligible_at`` -- the conventional
    "next bar open" assumption, and the default.

    Optimistic: the open print of bar N+1 lands at ``bar_end(N)``, a few seconds before
    the order could exist, so any edge inside those seconds is captured for free. It is
    nonetheless the closer estimate of a real fill, which lands a few seconds into bar
    N+1 rather than a full bar later, and its bias has a known direction and size. Every
    run using it records an ``optimistic_fill_rule`` warning.

    (The convention comes from daily-frequency research, where "fill at the next open"
    is unambiguously sound because hours separate the close from the open. At intraday
    frequency the same phrase makes a weaker claim -- hence the warning, and hence
    :attr:`NEXT_OPEN_AFTER_ELIGIBILITY` as the other side of the bracket.)"""

    NEXT_OPEN_AFTER_ELIGIBILITY = "next_open_after_eligibility"
    """Fill at the first bar open at or after ``eligible_at``, processed in a later clock
    step than the decision. With contiguous bars this is open(N+2) for a signal on bar N:
    bar N publishes no earlier than bar N+1's open, which has already printed.

    Conservative: it can never use a print from before the order existed, but it misses a
    full bar of the move, which erases a genuine one-bar edge entirely. Run it alongside
    the default; the pair brackets the truth."""


class SlippageKind(StrEnum):
    FIXED_BPS = "fixed_bps"
    PARTICIPATION = "participation"
    """``coefficient_bps * (quantity / liquidity)``: linear in participation."""

    SQRT_IMPACT = "sqrt_impact"
    """``coefficient * volatility * sqrt(quantity / liquidity)``: square-root impact."""


class SlippageConfig(FrozenModel):
    kind: SlippageKind = SlippageKind.PARTICIPATION
    coefficient: float = Field(default=25.0, ge=0)
    """bps for FIXED_BPS and PARTICIPATION (per 100% participation); dimensionless
    multiplier of volatility for SQRT_IMPACT."""

    fallback_bps: float = Field(default=10.0, ge=0)
    """Used when the model needs a liquidity or volatility estimate and none exists."""


class CostConfig(FrozenModel):
    half_spread_bps: float = Field(default=2.5, ge=0)
    """Assumed half-spread when no quote data exists. Labelled as an assumption."""

    commission_bps: float = Field(default=1.0, ge=0)
    fee_per_unit: float = Field(default=0.0, ge=0)
    min_commission: float = Field(default=0.0, ge=0)
    slippage: SlippageConfig = SlippageConfig()

    @classmethod
    def free(cls) -> CostConfig:
        """Zero costs. For tests and sensitivity baselines only."""
        return cls(
            half_spread_bps=0.0,
            commission_bps=0.0,
            slippage=SlippageConfig(kind=SlippageKind.FIXED_BPS, coefficient=0.0, fallback_bps=0.0),
        )


class MissingBarPolicy(StrEnum):
    WAIT = "wait"
    """Keep the order pending until the next open event or expiry."""


class ExecutionConfig(FrozenModel):
    submission_latency: _dt.timedelta = Field(default=_dt.timedelta(0), ge=_dt.timedelta(0))
    """``signal_at`` to ``order_at``: strategy compute plus transport to the gateway."""

    order_latency: _dt.timedelta = Field(default=_dt.timedelta(seconds=1), ge=_dt.timedelta(0))
    """``order_at`` to ``eligible_at``: gateway to venue."""

    fill_rule: FillRule = FillRule.OPEN_OF_CURRENT_BAR
    expire_after: _dt.timedelta | None = _dt.timedelta(minutes=5)
    """Unfilled orders expire this long after ``order_at``. None: never."""

    participation_cap: float | None = Field(default=0.1, gt=0)
    """Max fraction of the liquidity estimate that may fill at one open. None: no cap."""

    liquidity_lookback_bars: int = Field(default=20, ge=1)
    """Bars of trailing volume/volatility used for the estimates attached to each order."""

    missing_bar: MissingBarPolicy = MissingBarPolicy.WAIT
    costs: CostConfig = CostConfig()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.expire_after is not None and self.expire_after <= _dt.timedelta(0):
            raise ValueError("expire_after must be positive or None")
        return self


@dataclass(frozen=True, slots=True)
class OpenEvent:
    """The first print of a bar. All the execution model may see of that bar."""

    instrument_id: InstrumentId
    at: _dt.datetime
    open: float
    bar_start: _dt.datetime


@dataclass(slots=True)
class LiveOrder:
    """Mutable working state of an immutable :class:`Order`."""

    order: Order
    remaining: float
    filled: float = 0.0

    @property
    def order_id(self) -> OrderId:
        return self.order.order_id


@dataclass(frozen=True, slots=True)
class MatchResult:
    fills: list[Fill]
    notes: list[tuple[str, str]]
    """``(code, message)`` warnings raised while matching."""


class BarOpenExecutionModel:
    """Fill eligible market orders at an open event, subject to participation caps."""

    name = "bar_open:v1"

    def __init__(self, config: ExecutionConfig, *, quantity_increment: float) -> None:
        self.config = config
        self.increment = quantity_increment

    def match(
        self, event: OpenEvent, orders: list[LiveOrder], *, next_fill_id: int
    ) -> tuple[MatchResult, int]:
        fills: list[Fill] = []
        notes: list[tuple[str, str]] = []
        for live in orders:
            order = live.order
            assert order.instrument_id == event.instrument_id
            assert order.eligible_at <= event.at, "engine offered an ineligible order"

            quantity, note = self._fillable(live)
            if note is not None:
                notes.append(note)
            if quantity <= 0:
                continue

            reference = event.open
            half_spread = reference * self.config.costs.half_spread_bps / 1e4
            slippage, slip_note = self._slippage(reference, quantity, order)
            if slip_note is not None:
                notes.append(slip_note)
            price = reference + order.side.sign * (half_spread + slippage)
            fee = self._fee(quantity, price)
            participation = (
                quantity / order.liquidity_estimate if order.liquidity_estimate else None
            )
            fills.append(
                Fill(
                    fill_id=f"f{next_fill_id:08d}",
                    order_id=order.order_id,
                    instrument_id=order.instrument_id,
                    side=order.side,
                    quantity=quantity,
                    fill_at=event.at,
                    reference_price=reference,
                    half_spread=half_spread,
                    slippage=slippage,
                    price=price,
                    fee=fee,
                    execution_model=self.name,
                    participation=participation,
                )
            )
            next_fill_id += 1
            live.remaining -= quantity
            live.filled += quantity
        return MatchResult(fills=fills, notes=notes), next_fill_id

    def _fillable(self, live: LiveOrder) -> tuple[float, tuple[str, str] | None]:
        cap = self.config.participation_cap
        estimate = live.order.liquidity_estimate
        if cap is None:
            return live.remaining, None
        if estimate is None:
            return live.remaining, (
                "no_liquidity_estimate",
                f"order {live.order_id}: no trailing volume estimate; "
                "participation cap not applied",
            )
        allowed = _floor_to_increment(cap * estimate, self.increment)
        if allowed <= 0:
            return 0.0, (
                "participation_zero",
                f"order {live.order_id}: cap {cap} x estimate {estimate} is below one "
                f"increment ({self.increment}); nothing fillable at this open",
            )
        if allowed < live.remaining:
            return allowed, (
                "participation_limited",
                f"order {live.order_id}: {allowed} of {live.remaining} fillable at this open",
            )
        return live.remaining, None

    def _slippage(
        self, reference: float, quantity: float, order: Order
    ) -> tuple[float, tuple[str, str] | None]:
        cfg = self.config.costs.slippage
        match cfg.kind:
            case SlippageKind.FIXED_BPS:
                return reference * cfg.coefficient / 1e4, None
            case SlippageKind.PARTICIPATION:
                if not order.liquidity_estimate:
                    return reference * cfg.fallback_bps / 1e4, (
                        "slippage_fallback",
                        f"order {order.order_id}: no liquidity estimate; "
                        f"fallback {cfg.fallback_bps} bps",
                    )
                return reference * cfg.coefficient / 1e4 * (
                    quantity / order.liquidity_estimate
                ), None
            case SlippageKind.SQRT_IMPACT:
                if not order.liquidity_estimate or order.volatility_estimate is None:
                    return reference * cfg.fallback_bps / 1e4, (
                        "slippage_fallback",
                        f"order {order.order_id}: no liquidity/volatility estimate; "
                        f"fallback {cfg.fallback_bps} bps",
                    )
                return (
                    reference
                    * cfg.coefficient
                    * order.volatility_estimate
                    * math.sqrt(quantity / order.liquidity_estimate),
                    None,
                )

    def _fee(self, quantity: float, price: float) -> float:
        costs = self.config.costs
        fee = quantity * price * costs.commission_bps / 1e4 + quantity * costs.fee_per_unit
        return max(fee, costs.min_commission) if fee > 0 or costs.min_commission > 0 else 0.0


def _floor_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        return value
    steps = math.floor(value / increment + 1e-12)
    return steps * increment
