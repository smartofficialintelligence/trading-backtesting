"""Mean reversion that holds until the stretch is gone.

The rule strategy is stateless by design: it holds a position only while its entry
condition is still true. A 3-sigma fade written as a rule therefore exits the moment price
is 2.99 sigma away -- at the start of the reversion it was betting on, not the end. On the
six-month study that capped the gross edge at ~6 bps per round trip, with a median hold of
ten minutes (D69).

This strategy enters on a stretch and exits when the z-score has come back through
``exit_z``. It keeps no memory of its own: what it holds is read from the decision
context's positions and pending orders, so it cannot disagree with the portfolio about
whether it is in a trade -- a partial fill, an expired order, or a fold reset is simply
what the next decision sees.

While a position is held and not exiting, it emits **nothing**. Re-issuing the target
weight every bar would rebalance a drifting position back to size five minutes at a time,
and each of those small trades pays a commission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from qresearch.ids import InstrumentId
from qresearch.simulation.events import OrderIntent
from qresearch.strategy.contracts import DecisionContext


@dataclass
class ZScoreReversion:
    """Fade stretches in a z-score and hold until price is back near its mean."""

    zscore: str
    """Feature column holding the z-score, e.g. ``zscore_72``."""

    entry_z: float = 3.0
    """Enter long at ``z <= -entry_z``, short at ``z >= entry_z``."""

    exit_z: float = 0.0
    """Exit a long once ``z >= -exit_z``, a short once ``z <= exit_z``. 0 means at the mean."""

    sides: Literal["long", "short", "both"] = "both"
    rank_feature: str | None = None
    """Entries need ``rank_feature >= min_rank``; also breaks ties for scarce slots."""

    min_rank: float = 0.0
    trend_feature: str | None = None
    """Entries must go with this feature's sign: a long needs it above ``min_trend``, a short
    below ``-min_trend``. Meant for a longer trend, such as a basket's 30-day return (D70)."""

    min_trend: float = 0.0
    weight: float = 0.2
    """Target weight per position, as a fraction of equity."""

    max_positions: int | None = None

    strategy_id = "zscore_reversion:v1"

    def __post_init__(self) -> None:
        if self.entry_z <= 0:
            raise ValueError(f"entry_z must be positive, got {self.entry_z}")
        if self.exit_z >= self.entry_z:
            raise ValueError(
                f"exit_z ({self.exit_z}) must be below entry_z ({self.entry_z}), or a new "
                "position would qualify to exit on the bar it entered"
            )
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")
        if self.max_positions is not None and self.max_positions < 1:
            raise ValueError(f"max_positions must be at least 1, got {self.max_positions}")
        if self.rank_feature is None and self.min_rank != 0.0:
            raise ValueError("min_rank filters on rank_feature, which is not set")
        if self.trend_feature is None and self.min_trend != 0.0:
            raise ValueError("min_trend filters on trend_feature, which is not set")

    @property
    def required_features(self) -> set[str]:
        return {self.zscore} | {f for f in (self.rank_feature, self.trend_feature) if f}

    def reset(self) -> None:
        return None

    def on_decision(self, context: DecisionContext) -> Sequence[OrderIntent]:
        rows = _rows(context)
        pending = {o.instrument_id for o in context.pending_orders}
        held = {i: q for i, q in context.positions.items() if q != 0.0}
        intents: list[OrderIntent] = []

        staying = 0
        for instrument_id, quantity in sorted(held.items()):
            # Without a mark the exit cannot be sized; the position stays and retries next bar.
            if self._should_exit(quantity, rows.get(instrument_id)) and (
                instrument_id in context.marks
            ):
                intents.append(
                    OrderIntent.target_weight(
                        instrument_id, 0.0, at=context.decision_at, tag="exit"
                    )
                )
                continue
            staying += 1

        # A flat instrument with a live order is an entry in flight: it holds a slot and
        # must not be ordered again.
        in_flight = sum(1 for i in pending if i not in held)
        slots = None if self.max_positions is None else self.max_positions - staying - in_flight
        if slots is not None and slots <= 0:
            return intents

        candidates = []
        for instrument_id, row in rows.items():
            if instrument_id in held or instrument_id in pending:
                continue
            if instrument_id not in context.marks:
                continue
            direction = self._entry(row)
            if direction != 0.0:
                candidates.append((instrument_id, direction, row))
        candidates.sort(key=self._priority)
        for instrument_id, direction, _ in candidates[:slots]:
            intents.append(
                OrderIntent.target_weight(
                    instrument_id, direction * self.weight, at=context.decision_at, tag="entry"
                )
            )
        return intents

    def _entry(self, row: Mapping[str, Any]) -> float:
        z = _number(row.get(self.zscore))
        if z is None:
            return 0.0
        if self.rank_feature is not None:
            rank = _number(row.get(self.rank_feature))
            if rank is None or rank < self.min_rank:
                return 0.0
        direction = 0.0
        if z <= -self.entry_z and self.sides in ("long", "both"):
            direction = 1.0
        elif z >= self.entry_z and self.sides in ("short", "both"):
            direction = -1.0
        if direction == 0.0 or self.trend_feature is None:
            return direction
        # A missing trend blocks the entry: an unknown trend is not a trend in our favour.
        trend = _number(row.get(self.trend_feature))
        if trend is None or trend * direction <= self.min_trend:
            return 0.0
        return direction

    def _should_exit(self, quantity: float, row: Mapping[str, Any] | None) -> bool:
        """A missing z-score holds: no information is not a signal to trade."""
        z = None if row is None else _number(row.get(self.zscore))
        if z is None:
            return False
        return z >= -self.exit_z if quantity > 0 else z <= self.exit_z

    def _priority(self, candidate: tuple[InstrumentId, float, Mapping[str, Any]]) -> Any:
        """Highest rank first, then the largest stretch, then instrument id for determinism."""
        instrument_id, _, row = candidate
        rank = _number(row.get(self.rank_feature)) if self.rank_feature else None
        stretch = abs(_number(row.get(self.zscore)) or 0.0)
        return (-(rank if rank is not None else float("-inf")), -stretch, instrument_id)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _rows(context: DecisionContext) -> dict[InstrumentId, dict[str, Any]]:
    frame = context.features_latest
    if frame is None or frame.is_empty():
        return {}
    return {InstrumentId(str(r["instrument_id"])): r for r in frame.iter_rows(named=True)}
