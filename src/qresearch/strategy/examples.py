"""Small strategies used as fixtures and demonstrations. Not research."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from qresearch.simulation.events import OrderIntent
from qresearch.strategy.contracts import DecisionContext


@dataclass
class BuyAndHold:
    """Set target weights at the first decision and never trade again."""

    weights: Mapping[str, float]
    _done: bool = field(default=False, init=False)

    strategy_id = "buy_and_hold:v1"

    def reset(self) -> None:
        self._done = False

    def on_decision(self, context: DecisionContext) -> Sequence[OrderIntent]:
        if self._done:
            return ()
        intents = [
            OrderIntent.target_weight(i, w, at=context.decision_at, tag="initial")
            for i, w in sorted(self.weights.items())
            if i in context.marks
        ]
        if len(intents) == len(self.weights):
            self._done = True
        return intents


@dataclass
class ScheduledRebalance:
    """Re-issue target weights at the first decision and every ``every`` thereafter."""

    weights: Mapping[str, float]
    every: _dt.timedelta
    _last: _dt.datetime | None = field(default=None, init=False)

    strategy_id = "scheduled_rebalance:v1"

    def reset(self) -> None:
        self._last = None

    def on_decision(self, context: DecisionContext) -> Sequence[OrderIntent]:
        if self._last is not None and context.decision_at - self._last < self.every:
            return ()
        self._last = context.decision_at
        return [
            OrderIntent.target_weight(i, w, at=context.decision_at, tag="rebalance")
            for i, w in sorted(self.weights.items())
            if i in context.marks
        ]


@dataclass
class LaggedSignal:
    """Long ``weight`` in each instrument whose ``feature`` exceeds ``threshold``, else flat.

    The reference example of a feature-driven strategy. Because the feature row carries
    its own availability, the engine guarantees the value used here was knowable at
    ``decision_at``.
    """

    feature: str
    weight: float = 0.5
    threshold: float = 0.0
    instruments: tuple[str, ...] | None = None

    strategy_id = "lagged_signal:v1"

    def reset(self) -> None:
        return None

    def on_decision(self, context: DecisionContext) -> Sequence[OrderIntent]:
        universe = self.instruments or context.instruments
        intents = []
        for instrument_id in sorted(universe):
            value = context.feature(instrument_id, self.feature)
            if value is None or instrument_id not in context.marks:
                continue
            target = self.weight if value > self.threshold else 0.0
            if target == 0.0 and context.position(instrument_id) == 0.0:
                continue
            intents.append(OrderIntent.target_weight(instrument_id, target, at=context.decision_at))
        return intents
