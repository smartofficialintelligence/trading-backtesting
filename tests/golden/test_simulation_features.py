"""Features drive decisions only when they are available."""

from __future__ import annotations

import datetime as dt

import polars as pl

from qresearch.features.pipeline import compute_features
from qresearch.features.technical import LaggedReturn
from qresearch.research.splits import TimeRange
from qresearch.simulation.engine import run_simulation
from qresearch.strategy.contracts import DecisionContext
from qresearch.strategy.examples import BuyAndHold, LaggedSignal
from tests.golden.conftest import at, bars, free_config, instrument

# Closes alternate: up on even bars, down on odd bars.
ZIGZAG = [(m, 100.0, 101.0 if m % 2 == 0 else 99.0, 100.0) for m in range(8)]


class Recording(LaggedSignal):
    """LaggedSignal that also records what it saw."""

    def __init__(self) -> None:
        super().__init__(feature="ret_1", weight=0.5)
        self.seen: list[tuple[dt.datetime, object]] = []

    def on_decision(self, context: DecisionContext) -> list:  # type: ignore[override]
        self.seen.append((context.decision_at, context.feature("X", "ret_1")))
        return list(super().on_decision(context))


def test_the_signal_seen_at_each_decision_is_the_one_available_by_then() -> None:
    frame = bars("X", ZIGZAG)
    features = compute_features(frame, [LaggedReturn(1)])
    strategy = Recording()
    run_simulation(
        frame,
        strategy=strategy,
        instruments={"X": instrument("X")},
        config=free_config(),
        decision_range=TimeRange(start=at(0), end=at(100)),
        features=features,
    )
    expected = {row["available_at"]: row["ret_1"] for row in features.iter_rows(named=True)}
    for decision_at, value in strategy.seen:
        assert value == expected[decision_at]


def test_positions_follow_the_signal_with_the_execution_delay() -> None:
    """ret_1 > 0 after even bars (close 101 > previous 99). A decision on bar 2 (published
    00:03:02) goes long; the fill lands at bar 4's open (00:04:00)."""
    frame = bars("X", ZIGZAG)
    features = compute_features(frame, [LaggedReturn(1)])
    result = run_simulation(
        frame,
        strategy=LaggedSignal("ret_1", weight=0.5),
        instruments={"X": instrument("X")},
        config=free_config(),
        decision_range=TimeRange(start=at(0), end=at(100)),
        features=features,
    )
    buys = [o for o in result.orders if o.side.value == "buy"]
    assert buys and buys[0].signal_at == at(3, 2)
    assert result.fills[0].fill_at == at(4, 0)


def test_a_late_feature_row_is_not_seen_until_it_publishes() -> None:
    frame = bars("X", ZIGZAG, late={2: dt.timedelta(minutes=3)})  # bar 2 publishes 00:06:02
    features = compute_features(frame, [LaggedReturn(1)])
    strategy = Recording()
    run_simulation(
        frame,
        strategy=strategy,
        instruments={"X": instrument("X")},
        config=free_config(),
        decision_range=TimeRange(start=at(0), end=at(100)),
        features=features,
    )
    at_3 = next(v for t, v in strategy.seen if t == at(4, 2))
    # At 00:04:02 bar 3 has published but bar 2 has not, so ret_1 for bar 3 (needs bar 2)
    # is not available; the latest published feature row is bar 1's.
    assert at_3 == pl.DataFrame(features.filter(pl.col("bar_start") == at(1))).item(0, "ret_1")


def test_buy_and_hold_trades_once() -> None:
    frame = bars("X", ZIGZAG)
    result = run_simulation(
        frame,
        strategy=BuyAndHold({"X": 0.5}),
        instruments={"X": instrument("X")},
        config=free_config(),
        decision_range=TimeRange(start=at(0), end=at(100)),
    )
    assert len(result.orders) == 1
    assert result.orders[0].tag == "initial"
    assert result.final.position_count == 1
