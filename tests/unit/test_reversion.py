"""Z-score reversion: entry, holding without churn, exit, and slot accounting."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl
import pytest

from qresearch.introspect import component_catalog
from qresearch.strategy.registry import build_strategy
from qresearch.strategy.reversion import ZScoreReversion

T0 = dt.datetime(2024, 3, 4, tzinfo=dt.UTC)


@dataclass
class _Order:
    instrument_id: str


class _Context:
    """Minimal DecisionContext stand-in."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        positions: dict[str, float] | None = None,
        pending: tuple[str, ...] = (),
    ) -> None:
        self.decision_at = T0
        self.features_latest = pl.DataFrame(rows) if rows else None
        self.positions = positions or {}
        self.marks = {str(r["instrument_id"]): 100.0 for r in rows}
        self.pending_orders = tuple(_Order(i) for i in pending)


def rows(*items: tuple[str, float | None, float]) -> list[dict[str, object]]:
    """``(instrument, zscore, rank)``."""
    return [{"instrument_id": i, "z": z, "rank": r} for i, z, r in items]


def targets(strategy: ZScoreReversion, context: _Context) -> dict[str, float]:
    return {i.instrument_id: i.value for i in strategy.on_decision(context)}  # type: ignore[arg-type]


# -- entry ----------------------------------------------------------------------------------


def test_a_stretch_below_entry_opens_a_long_and_above_opens_a_short() -> None:
    strategy = ZScoreReversion(zscore="z", entry_z=3.0, weight=0.2)
    context = _Context(rows(("A", -3.5, 1.0), ("B", 3.1, 1.0), ("C", -2.9, 1.0)))
    assert targets(strategy, context) == {"A": 0.2, "B": -0.2}


def test_long_only_never_shorts() -> None:
    strategy = ZScoreReversion(zscore="z", sides="long")
    assert targets(strategy, _Context(rows(("A", -4.0, 1.0), ("B", 4.0, 1.0)))) == {"A": 0.2}


def test_entries_need_the_rank_filter() -> None:
    strategy = ZScoreReversion(zscore="z", rank_feature="rank", min_rank=0.8)
    context = _Context(rows(("A", -4.0, 0.9), ("B", -4.0, 0.5), ("C", -4.0, None)))  # type: ignore[arg-type]
    assert targets(strategy, context) == {"A": 0.2}


# -- holding and exit -----------------------------------------------------------------------


def test_a_position_between_entry_and_exit_is_left_alone() -> None:
    """The point of the strategy: no exit at 2.99 sigma, and no rebalancing trades either."""
    strategy = ZScoreReversion(zscore="z", entry_z=3.0, exit_z=0.0)
    context = _Context(rows(("A", -1.0, 1.0)), positions={"A": 20.0})
    assert strategy.on_decision(context) == []


def test_a_long_exits_once_back_at_the_mean_and_a_short_symmetrically() -> None:
    strategy = ZScoreReversion(zscore="z", entry_z=3.0, exit_z=0.0)
    context = _Context(rows(("A", 0.1, 1.0), ("B", -0.1, 1.0)), positions={"A": 5.0, "B": -5.0})
    assert targets(strategy, context) == {"A": 0.0, "B": 0.0}


def test_exit_z_moves_the_exit_short_of_the_mean() -> None:
    strategy = ZScoreReversion(zscore="z", entry_z=3.0, exit_z=1.0)
    assert targets(strategy, _Context(rows(("A", -0.9, 1.0)), positions={"A": 5.0})) == {"A": 0.0}
    assert targets(strategy, _Context(rows(("A", -1.1, 1.0)), positions={"A": 5.0})) == {}


def test_a_missing_zscore_holds_rather_than_trades() -> None:
    strategy = ZScoreReversion(zscore="z")
    context = _Context(rows(("A", None, 1.0)), positions={"A": 5.0})
    assert strategy.on_decision(context) == []


def test_a_held_position_does_not_need_the_rank_filter_to_stay() -> None:
    """Rank gates entries only; a name that stops being volatile is still mid-reversion."""
    strategy = ZScoreReversion(zscore="z", rank_feature="rank", min_rank=0.8)
    context = _Context(rows(("A", -2.0, 0.1)), positions={"A": 5.0})
    assert strategy.on_decision(context) == []


# -- slots ------------------------------------------------------------------------------------


def test_held_positions_keep_their_slots() -> None:
    """A fresher stretch must not evict a trade that has not finished reverting."""
    strategy = ZScoreReversion(zscore="z", max_positions=1)
    context = _Context(rows(("A", -1.0, 1.0), ("B", -5.0, 1.0)), positions={"A": 5.0})
    assert strategy.on_decision(context) == []


def test_an_exit_frees_its_slot_on_the_same_decision() -> None:
    strategy = ZScoreReversion(zscore="z", max_positions=1)
    context = _Context(rows(("A", 0.5, 1.0), ("B", -5.0, 1.0)), positions={"A": 5.0})
    assert targets(strategy, context) == {"A": 0.0, "B": 0.2}


def test_an_entry_in_flight_is_not_reordered_and_holds_its_slot() -> None:
    strategy = ZScoreReversion(zscore="z", max_positions=1)
    context = _Context(rows(("A", -4.0, 1.0), ("B", -5.0, 1.0)), pending=("A",))
    assert strategy.on_decision(context) == []


def test_scarce_slots_go_to_rank_then_stretch_regardless_of_row_order() -> None:
    strategy = ZScoreReversion(zscore="z", rank_feature="rank", max_positions=2)
    items = (("A", -3.5, 0.9), ("B", -6.0, 0.9), ("C", -9.0, 0.5), ("D", -3.2, 1.0))
    forward = strategy.on_decision(_Context(rows(*items)))
    reverse = strategy.on_decision(_Context(rows(*reversed(items))))
    assert [i.instrument_id for i in forward] == [i.instrument_id for i in reverse] == ["D", "B"]


def test_an_instrument_without_a_mark_is_neither_entered_nor_exited() -> None:
    strategy = ZScoreReversion(zscore="z", max_positions=1)
    context = _Context(rows(("A", 0.5, 1.0), ("B", -5.0, 1.0)), positions={"A": 5.0})
    context.marks = {}
    assert strategy.on_decision(context) == []


# -- construction ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"entry_z": 0.0}, "entry_z must be positive"),
        ({"entry_z": 2.0, "exit_z": 2.0}, "must be below entry_z"),
        ({"weight": 0.0}, "weight must be positive"),
        ({"max_positions": 0}, "at least 1"),
        ({"min_rank": 0.5}, "rank_feature"),
    ],
)
def test_invalid_parameters_are_refused(params: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ZScoreReversion(zscore="z", **params)  # type: ignore[arg-type]


def test_required_features_lets_the_orchestrator_catch_a_typo() -> None:
    assert ZScoreReversion(zscore="z").required_features == {"z"}
    assert ZScoreReversion(zscore="z", rank_feature="r").required_features == {"z", "r"}


def test_built_from_yaml_shaped_params() -> None:
    strategy = build_strategy(
        "zscore_reversion",
        {"zscore": "zscore_72", "entry_z": "3", "sides": "long", "max_positions": 2},
    )
    assert isinstance(strategy, ZScoreReversion)
    assert strategy.entry_z == 3.0 and strategy.sides == "long" and strategy.max_positions == 2


def test_the_ui_can_render_a_form_for_it() -> None:
    spec = next(s for s in component_catalog().strategies if s.kind == "zscore_reversion")
    by_name = {p.name: p for p in spec.params}
    assert by_name["sides"].type == "choice"
    assert by_name["sides"].choices == ("long", "short", "both")
    assert by_name["rank_feature"].optional
    assert all(p.type != "unsupported" for p in spec.params)
