"""Two assets: simultaneous events, asynchrony, and permutation invariance."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.golden.conftest import Scripted, at, bars, free_config, instrument
from tests.golden.test_simulation_timing import FOUR

from qresearch.research.splits import TimeRange
from qresearch.simulation.engine import run_simulation

FOUR_Y = [
    (0, 50.0, 51.0, 500.0),
    (1, 52.0, 53.0, 500.0),
    (2, 54.0, 55.0, 500.0),
    (3, 56.0, 57.0, 500.0),
]


def two_assets(**kwargs: object) -> pl.DataFrame:
    return pl.concat([bars("X", FOUR), bars("Y", FOUR_Y, **kwargs)])  # type: ignore[arg-type]


def test_simultaneous_orders_fill_at_each_instruments_own_open() -> None:
    result = run_simulation(
        two_assets(),
        strategy=Scripted({1: [("delta", "X", 10.0), ("delta", "Y", 20.0)]}),
        instruments={"X": instrument("X"), "Y": instrument("Y")},
        config=free_config(),
        decision_range=TimeRange(start=at(0), end=at(100)),
    )
    by_instrument = {f.instrument_id: f for f in result.fills}
    assert by_instrument["X"].price == 104.0 and by_instrument["Y"].price == 54.0
    assert by_instrument["X"].fill_at == by_instrument["Y"].fill_at == at(2, 0)
    after = next(s for s in result.snapshots if s.at == at(2, 0))
    assert after.cash == pytest.approx(100_000 - 1040 - 1080)
    assert result.final.equity == pytest.approx(100_000 + 10 * (107 - 104) + 20 * (57 - 54))


def test_permuting_rows_intents_and_instrument_order_changes_nothing() -> None:
    def go(
        frame: pl.DataFrame, intents: list[tuple[str, str, float]], ids: list[str]
    ) -> dict[str, pl.DataFrame]:
        return run_simulation(
            frame,
            strategy=Scripted({1: intents, 2: [("target_weight", i, 0.2) for i in ids]}),
            instruments={i: instrument(i) for i in ids},
            config=free_config(),
            decision_range=TimeRange(start=at(0), end=at(100)),
        ).frames()

    base = go(two_assets(), [("delta", "X", 10.0), ("delta", "Y", 20.0)], ["X", "Y"])
    permuted = go(
        two_assets().sample(fraction=1.0, shuffle=True, seed=3),
        [("delta", "Y", 20.0), ("delta", "X", 10.0)],
        ["Y", "X"],
    )
    for name in ("orders", "fills", "equity_curve", "positions", "intent_outcomes"):
        assert base[name].equals(permuted[name]), name


def test_a_late_bar_for_one_asset_produces_its_own_decision_instant() -> None:
    """Y's bar 1 publishes at 00:02:30 instead of 00:02:02. The strategy is invoked at
    00:02:02 with X's bar 1 and Y's bar 0, then again at 00:02:30 with Y's bar 1."""
    strategy = Scripted({})
    run_simulation(
        two_assets(late={1: dt.timedelta(seconds=28)}),
        strategy=strategy,
        instruments={"X": instrument("X"), "Y": instrument("Y")},
        config=free_config(),
        decision_range=TimeRange(start=at(0), end=at(100)),
    )
    instants = [c.decision_at for c in strategy.seen]
    assert at(2, 2) in instants and at(2, 30) in instants
    early = next(c for c in strategy.seen if c.decision_at == at(2, 2))
    latest = dict(
        zip(early.bars_latest["instrument_id"], early.bars_latest["bar_start"], strict=True)
    )
    assert latest["X"] == at(1) and latest["Y"] == at(0)
    assert early.marks["Y"] == 51.0, "Y's mark is still bar 0's close"


def test_an_instrument_without_a_definition_is_refused() -> None:
    with pytest.raises(ValueError, match="without definitions"):
        run_simulation(
            two_assets(),
            strategy=Scripted({}),
            instruments={"X": instrument("X")},
            config=free_config(),
            decision_range=TimeRange(start=at(0), end=at(100)),
        )
