"""The two fill rules bracket the truth by exactly one bar's move on each signal fill.

D14 in docs/decisions.md: a decision on bar N fills at open(N+2) under the conservative
rule and at open(N+1) under the textbook rule. For a single order, nothing else differs --
so the equity gap between the two runs is ``quantity x (open(N+2) - open(N+1))`` to the
cent, at every mark thereafter. This pins the size of the assumption so it can be read as
a number rather than a worry.
"""

from __future__ import annotations

import datetime as dt

import pytest
from tests.golden.conftest import Scripted, bars, free_config, run
from tests.golden.test_simulation_timing import FOUR

from qresearch.simulation.execution import ExecutionConfig, FillRule


def _config(rule: FillRule) -> object:
    return free_config(
        execution=ExecutionConfig(
            order_latency=dt.timedelta(0),
            expire_after=None,
            participation_cap=None,
            liquidity_lookback_bars=1,
            fill_rule=rule,
            costs=free_config().execution.costs,
        )
    )


def test_the_bracket_is_exactly_one_bars_open_to_open_move() -> None:
    conservative = run(
        bars("X", FOUR),
        Scripted({1: [("delta", "X", 10.0)]}),
        config=_config(FillRule.NEXT_OPEN_AFTER_ELIGIBILITY),
    )  # type: ignore[arg-type]
    optimistic = run(
        bars("X", FOUR),
        Scripted({1: [("delta", "X", 10.0)]}),
        config=_config(FillRule.OPEN_OF_CURRENT_BAR),
    )  # type: ignore[arg-type]

    assert conservative.fills[0].price == 104.0, "open of bar 2"
    assert optimistic.fills[0].price == 102.0, "open of bar 1"
    gap = 10.0 * (104.0 - 102.0)
    assert optimistic.final.equity - conservative.final.equity == pytest.approx(gap)

    # Every snapshot after both fills shows the same gap: the rules differ in the fill and
    # in nothing else.
    later = {
        s.at: s.equity for s in conservative.snapshots if s.at >= conservative.fills[0].fill_at
    }
    for at, equity in later.items():
        match = next((s for s in optimistic.snapshots if s.at == at), None)
        if match is not None:
            assert match.equity - equity == pytest.approx(gap)


def test_a_one_bar_edge_is_visible_only_under_the_optimistic_rule() -> None:
    """The consequence that matters for research: with perfect one-bar foresight, the
    conservative rule shows nothing and the optimistic rule shows the whole move."""
    # Bar 1 rallies (open 102 -> close 103), bar 2 is flat, bar 3 is flat.
    frame = bars(
        "X",
        [
            (0, 100.0, 101.0, 1.0),
            (1, 102.0, 110.0, 1.0),
            (2, 110.0, 110.0, 1.0),
            (3, 110.0, 110.0, 1.0),
        ],
    )
    conservative = run(
        frame,
        Scripted({1: [("delta", "X", 10.0)]}),
        config=_config(FillRule.NEXT_OPEN_AFTER_ELIGIBILITY),
    )  # type: ignore[arg-type]
    optimistic = run(
        frame, Scripted({1: [("delta", "X", 10.0)]}), config=_config(FillRule.OPEN_OF_CURRENT_BAR)
    )  # type: ignore[arg-type]
    assert conservative.final.equity == pytest.approx(100_000.0), "filled after the move, at 110"
    assert optimistic.final.equity == pytest.approx(100_000.0 + 10 * (110.0 - 102.0)), (
        "filled before it, at 102"
    )
