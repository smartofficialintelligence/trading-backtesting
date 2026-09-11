"""Intent sizing and portfolio constraints.

Turns what a strategy *asked for* into what the portfolio *may do*, and records every
change of mind (ARCHITECTURE.md sec. 6, ``ConstraintSet``: "every rejection or resize is
recorded").

Two properties matter beyond the rules themselves:

* **Order independence.** Constraints that bind across instruments (gross exposure, cash)
  are applied by scaling every exposure-increasing order by one common factor, never by
  processing instruments in sequence and cutting whichever came last. Permuting the
  strategy's intents cannot change the economic result.
* **Pending orders count.** A target is measured against the current position *plus*
  unfilled orders, so a strategy that repeats a target while its order is still in flight
  does not double up.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import Field

from qresearch.config import FrozenModel
from qresearch.ids import InstrumentId
from qresearch.simulation.events import IntentKind, IntentOutcome, OrderIntent, RejectionReason


class ConstraintConfig(FrozenModel):
    allow_short: bool = False
    max_gross_exposure: float = Field(default=1.0, gt=0)
    """Sum of |position value| after the batch, as a multiple of equity."""

    max_position_weight: float = Field(default=1.0, gt=0)
    """|position value| per instrument after the batch, as a fraction of equity."""

    cost_buffer_bps: float = Field(default=10.0, ge=0)
    """Estimated friction reserved from cash when checking buys can be paid for."""


@dataclass(frozen=True, slots=True)
class SizingState:
    """Everything sizing needs, captured at the decision instant."""

    equity: float
    cash: float
    marks: Mapping[InstrumentId, float]
    positions: Mapping[InstrumentId, float]
    pending: Mapping[InstrumentId, float]
    """Signed unfilled quantity per instrument."""

    increments: Mapping[InstrumentId, float]

    def base(self, instrument_id: InstrumentId) -> float:
        return self.positions.get(instrument_id, 0.0) + self.pending.get(instrument_id, 0.0)


@dataclass(slots=True)
class _Working:
    intent: OrderIntent
    requested: float
    quantity: float
    rejection: RejectionReason | None = None
    notes: list[str] | None = None

    def note(self, text: str) -> None:
        if self.notes is None:
            self.notes = []
        self.notes.append(text)


def floor_to_increment(value: float, increment: float) -> float:
    """Round a signed quantity toward zero to a multiple of ``increment``."""
    if increment <= 0:
        return value
    steps = math.floor(abs(value) / increment + 1e-9)
    return math.copysign(steps * increment, value) if steps else 0.0


def size_intents(
    intents: Sequence[OrderIntent],
    state: SizingState,
    config: ConstraintConfig,
    *,
    at: _dt.datetime,
) -> list[IntentOutcome]:
    """Convert intents to signed order quantities under the constraint set.

    Returns one outcome per intent, in the order given. ``final_quantity`` is signed and a
    multiple of the instrument's increment; zero means no order.
    """
    working = [_request(intent, state, at) for intent in intents]
    _apply_increment_and_short(working, state, config)
    _apply_position_limit(working, state, config)
    _apply_gross_limit(working, state, config)
    _apply_cash(working, state, config)
    return [
        IntentOutcome(
            at=at,
            instrument_id=w.intent.instrument_id,
            intent_kind=w.intent.kind,
            intent_value=w.intent.value,
            requested_quantity=w.requested,
            final_quantity=w.quantity,
            rejection=w.rejection,
            note="; ".join(w.notes) if w.notes else None,
        )
        for w in working
    ]


def _request(intent: OrderIntent, state: SizingState, at: _dt.datetime) -> _Working:
    instrument_id = intent.instrument_id
    if intent.created_at != at:
        w = _Working(intent, 0.0, 0.0, RejectionReason.STALE_INTENT)
        w.note(f"intent created at {intent.created_at}, decision is at {at}")
        return w
    if instrument_id not in state.increments:
        return _Working(intent, 0.0, 0.0, RejectionReason.UNKNOWN_INSTRUMENT)
    match intent.kind:
        case IntentKind.DELTA:
            requested = intent.value
        case IntentKind.TARGET_QUANTITY:
            requested = intent.value - state.base(instrument_id)
        case IntentKind.TARGET_WEIGHT:
            mark = state.marks.get(instrument_id)
            if mark is None or mark <= 0:
                w = _Working(intent, 0.0, 0.0, RejectionReason.NO_MARK)
                w.note("target weight needs a point-in-time mark and none is available")
                return w
            requested = intent.value * state.equity / mark - state.base(instrument_id)
    return _Working(intent, requested, requested)


def _apply_increment_and_short(
    working: list[_Working], state: SizingState, config: ConstraintConfig
) -> None:
    for w in working:
        if w.rejection is not None:
            continue
        instrument_id = w.intent.instrument_id
        increment = state.increments[instrument_id]
        base = state.base(instrument_id)
        if not config.allow_short and base + w.quantity < 0:
            resized = -base
            w.note(f"shorting not allowed: resized {w.quantity:.10g} to {resized:.10g}")
            w.quantity = resized
        rounded = floor_to_increment(w.quantity, increment)
        if rounded == 0.0:
            if w.requested != 0.0:
                w.rejection = (
                    RejectionReason.SHORT_NOT_ALLOWED
                    if w.quantity == 0.0 and w.requested < 0 and not config.allow_short
                    else RejectionReason.BELOW_INCREMENT
                )
            w.quantity = 0.0
            continue
        if rounded != w.quantity:
            w.note(f"rounded {w.quantity:.10g} to increment {increment}: {rounded:.10g}")
        w.quantity = rounded


def _apply_position_limit(
    working: list[_Working], state: SizingState, config: ConstraintConfig
) -> None:
    limit_value = config.max_position_weight * state.equity
    for w in working:
        if w.rejection is not None or w.quantity == 0.0:
            continue
        instrument_id = w.intent.instrument_id
        mark = state.marks.get(instrument_id)
        if mark is None:
            continue
        base = state.base(instrument_id)
        projected = base + w.quantity
        if abs(projected) <= abs(base) or abs(projected) * mark <= limit_value + 1e-9:
            continue
        allowed_abs = limit_value / mark
        target = math.copysign(allowed_abs, projected)
        new_quantity = floor_to_increment(target - base, state.increments[instrument_id])
        if new_quantity == 0.0 or (new_quantity > 0) != (w.quantity > 0):
            w.rejection = RejectionReason.POSITION_LIMIT
            w.note(f"position limit {config.max_position_weight:g} x equity leaves no room")
            w.quantity = 0.0
        else:
            w.note(f"position limit: resized {w.quantity:.10g} to {new_quantity:.10g}")
            w.quantity = new_quantity


def _apply_gross_limit(
    working: list[_Working], state: SizingState, config: ConstraintConfig
) -> None:
    limit = config.max_gross_exposure * state.equity
    active = [w for w in working if w.rejection is None and w.quantity != 0.0]
    touched = {w.intent.instrument_id for w in active}
    untouched = sum(
        abs(q * state.marks[i])
        for i, q in state.positions.items()
        if i not in touched and i in state.marks
    )
    untouched += sum(
        abs(q * state.marks[i])
        for i, q in state.pending.items()
        if i not in touched and i in state.marks and i not in state.positions
    )
    fixed = untouched
    increases: list[tuple[_Working, float, float]] = []
    for w in active:
        instrument_id = w.intent.instrument_id
        mark = state.marks.get(instrument_id)
        if mark is None:
            continue
        base = state.base(instrument_id)
        projected = base + w.quantity
        if abs(projected) > abs(base):
            fixed += abs(base) * mark
            increases.append((w, abs(base), (abs(projected) - abs(base)) * mark))
        else:
            fixed += abs(projected) * mark
    delta = sum(inc for _, _, inc in increases)
    if fixed + delta <= limit + 1e-9 or not increases:
        return
    factor = max(0.0, (limit - fixed) / delta) if delta > 0 else 0.0
    for w, base_abs, _ in increases:
        instrument_id = w.intent.instrument_id
        base = state.base(instrument_id)
        projected = base + w.quantity
        new_abs = base_abs + factor * (abs(projected) - base_abs)
        target = math.copysign(new_abs, projected)
        new_quantity = floor_to_increment(target - base, state.increments[instrument_id])
        if new_quantity == 0.0 or (new_quantity > 0) != (w.quantity > 0):
            w.rejection = RejectionReason.GROSS_EXPOSURE
            w.note(f"gross exposure limit {config.max_gross_exposure:g} x equity: no room")
            w.quantity = 0.0
        else:
            w.note(f"gross exposure limit: scaled {w.quantity:.10g} to {new_quantity:.10g}")
            w.quantity = new_quantity


def _apply_cash(working: list[_Working], state: SizingState, config: ConstraintConfig) -> None:
    buffer = 1.0 + config.cost_buffer_bps / 1e4
    buys = [w for w in working if w.rejection is None and w.quantity > 0]
    if not buys:
        return
    sells_value = sum(
        -w.quantity * state.marks[w.intent.instrument_id]
        for w in working
        if w.rejection is None and w.quantity < 0 and w.intent.instrument_id in state.marks
    )
    buy_cost = sum(
        w.quantity * state.marks[w.intent.instrument_id] * buffer
        for w in buys
        if w.intent.instrument_id in state.marks
    )
    available = state.cash + sells_value
    if buy_cost <= available + 1e-9:
        return
    factor = max(0.0, available / buy_cost) if buy_cost > 0 else 0.0
    for w in buys:
        new_quantity = floor_to_increment(
            w.quantity * factor, state.increments[w.intent.instrument_id]
        )
        if new_quantity <= 0.0:
            w.rejection = RejectionReason.INSUFFICIENT_CASH
            w.quantity = 0.0
        else:
            w.note(f"cash: scaled {w.quantity:.10g} to {new_quantity:.10g}")
            w.quantity = new_quantity
