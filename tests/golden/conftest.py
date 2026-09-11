"""Builders for hand-calculable simulation scenarios."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import polars as pl

from qresearch.data.contracts import AssetClass, Instrument
from qresearch.ids import InstrumentId
from qresearch.research.splits import TimeRange
from qresearch.simulation.engine import (
    EndOfRunPolicy,
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from qresearch.simulation.events import OrderIntent
from qresearch.simulation.execution import CostConfig, ExecutionConfig, FillRule
from qresearch.strategy.contracts import DecisionContext

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)
TS = pl.Datetime("us", "UTC")


def at(minute: float, second: float = 0) -> dt.datetime:
    return T0 + dt.timedelta(minutes=minute, seconds=second)


def instrument(instrument_id: str, *, increment: str = "1") -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(instrument_id),
        asset_class=AssetClass.CRYPTO,
        venue="TEST",
        quote_currency="USD",
        price_increment=Decimal("0.01"),
        quantity_increment=Decimal(increment),
        calendar_id="24x7:1",
    )


def bars(
    instrument_id: str,
    rows: Sequence[tuple[int, float, float, float]],
    *,
    latency: dt.timedelta = dt.timedelta(seconds=2),
    late: dict[int, dt.timedelta] | None = None,
) -> pl.DataFrame:
    """``(minute, open, close, volume)`` rows; high/low are derived to be valid."""
    late = late or {}
    records = []
    for minute, open_, close, volume in rows:
        start = at(minute)
        end = start + dt.timedelta(minutes=1)
        records.append(
            {
                "instrument_id": instrument_id,
                "bar_size": "1m",
                "bar_start": start,
                "bar_end": end,
                "available_at": end + latency + late.get(minute, dt.timedelta(0)),
                "open": open_,
                "high": max(open_, close) + 0.5,
                "low": min(open_, close) - 0.5,
                "close": close,
                "volume": volume,
                "vwap": None,
                "trade_count": None,
                "source": "golden",
                "source_key": None,
                "revision": 0,
                "ingested_at": T0,
            }
        )
    return pl.DataFrame(
        records,
        schema_overrides={
            "bar_start": TS,
            "bar_end": TS,
            "available_at": TS,
            "ingested_at": TS,
            "vwap": pl.Float64,
            "trade_count": pl.Int64,
            "revision": pl.Int32,
        },
    )


@dataclass
class Scripted:
    """Emit the intents scheduled for the n-th decision (1-based); nothing otherwise.

    Intents are built at decision time so ``created_at`` matches.
    """

    script: dict[int, list[tuple[str, str, float]]]
    """decision index -> [(kind, instrument, value)].

    ``kind`` is one of delta / target_quantity / target_weight.
    """

    seen: list[DecisionContext] = field(default_factory=list)
    strategy_id = "scripted:test"

    def reset(self) -> None:
        self.seen = []

    def on_decision(self, context: DecisionContext) -> list[OrderIntent]:
        self.seen.append(context)
        n = len(self.seen)
        return [
            getattr(OrderIntent, kind)(instrument_id, value, at=context.decision_at)
            for kind, instrument_id, value in self.script.get(n, [])
        ]


def free_config(**overrides: object) -> SimulationConfig:
    """No costs, no latency, no cap, no expiry: the cleanest possible timing scenario."""
    execution = ExecutionConfig(
        submission_latency=dt.timedelta(0),
        order_latency=dt.timedelta(0),
        expire_after=None,
        participation_cap=None,
        liquidity_lookback_bars=1,
        costs=CostConfig.free(),
    )
    base: dict[str, object] = {
        "initial_cash": 100_000.0,
        "execution": execution,
        "max_mark_staleness": None,
    }
    return SimulationConfig.model_validate(base | overrides)


def run(
    frame: pl.DataFrame,
    strategy: Scripted,
    *,
    config: SimulationConfig | None = None,
    instruments: dict[str, Instrument] | None = None,
    start: int = 0,
    end: int = 10_000,
    features: pl.DataFrame | None = None,
) -> SimulationResult:
    ids = sorted(frame.get_column("instrument_id").unique().to_list())
    return run_simulation(
        frame,
        strategy=strategy,
        instruments=instruments or {i: instrument(i) for i in ids},
        config=config or free_config(),
        decision_range=TimeRange(start=at(start), end=at(end)),
        features=features,
    )


__all__ = [
    "T0",
    "EndOfRunPolicy",
    "FillRule",
    "Scripted",
    "at",
    "bars",
    "free_config",
    "instrument",
    "run",
]
