"""What a strategy sees and what it returns.

The :class:`DecisionContext` is the whole of a strategy's view of the world at one
instant (ARCHITECTURE.md sec. 2, "strategy code cannot access the raw future"). It holds
the latest *published* bar and feature row per instrument, a short recent history if the
run asks for one, and the portfolio state. It does not hold a catalog, a full frame, the
execution model, or anything with an ``available_at`` later than ``decision_at``.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import polars as pl

from qresearch.ids import InstrumentId
from qresearch.simulation.events import Order, OrderIntent


@dataclass(frozen=True, slots=True)
class DecisionContext:
    decision_at: _dt.datetime
    instruments: tuple[InstrumentId, ...]
    """Instruments with at least one published bar by now, sorted."""

    bars_latest: pl.DataFrame
    """One row per instrument: its most recently published bar."""

    bars_recent: pl.DataFrame
    """The last ``context_lookback_bars`` published bars per instrument."""

    features_latest: pl.DataFrame | None
    """One row per instrument with published features, or None if the run has none."""

    features_recent: pl.DataFrame | None
    positions: Mapping[InstrumentId, float]
    cash: float
    equity: float
    marks: Mapping[InstrumentId, float]
    """Latest published close per instrument."""

    pending_orders: tuple[Order, ...]
    run_id: str | None = None
    fold: int | None = None

    def position(self, instrument_id: str) -> float:
        return self.positions.get(InstrumentId(instrument_id), 0.0)

    def feature(self, instrument_id: str, name: str) -> Any:
        """Latest value of feature ``name`` for ``instrument_id``, or None."""
        if self.features_latest is None or name not in self.features_latest.columns:
            return None
        rows = self.features_latest.filter(pl.col("instrument_id") == instrument_id)
        return None if rows.is_empty() else rows.item(0, name)

    def latest_bar(self, instrument_id: str) -> dict[str, Any] | None:
        rows = self.bars_latest.filter(pl.col("instrument_id") == instrument_id)
        return None if rows.is_empty() else rows.row(0, named=True)


class Strategy(Protocol):
    @property
    def strategy_id(self) -> str: ...

    def reset(self) -> None:
        """Clear any state. Called once before each fold/run."""
        ...

    def on_decision(self, context: DecisionContext) -> Sequence[OrderIntent]: ...
