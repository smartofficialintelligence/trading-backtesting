"""Name -> strategy registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from qresearch.features.registry import construct
from qresearch.strategy.contracts import Strategy
from qresearch.strategy.examples import BuyAndHold, LaggedSignal, ScheduledRebalance

STRATEGIES: Final[dict[str, type[Any]]] = {
    "buy_and_hold": BuyAndHold,
    "scheduled_rebalance": ScheduledRebalance,
    "lagged_signal": LaggedSignal,
}


def build_strategy(kind: str, params: Mapping[str, Any]) -> Strategy:
    try:
        cls = STRATEGIES[kind]
    except KeyError:
        raise KeyError(f"unknown strategy kind {kind!r}; known: {sorted(STRATEGIES)}") from None
    strategy = construct(cls, params)
    return strategy  # type: ignore[no-any-return]
