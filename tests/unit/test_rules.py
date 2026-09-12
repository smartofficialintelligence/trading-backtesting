"""Rule strategies: evaluation, selection, and the leakage properties they inherit."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from pydantic import ValidationError

from qresearch.strategy.rules import AllOf, AnyOf, Compare, Not, Operator, Rule, RuleStrategy

T0 = dt.datetime(2024, 3, 4, tzinfo=dt.UTC)


def compare(feature: str, op: str, value: float) -> Compare:
    return Compare(feature=feature, op=Operator(op), value=value)


# -- evaluation ---------------------------------------------------------------------------


def test_a_comparison_reads_the_named_feature() -> None:
    condition = compare("zscore_20", "<", -1.5)
    assert condition.evaluate({"zscore_20": -2.0}) is True
    assert condition.evaluate({"zscore_20": -1.0}) is False
    assert condition.evaluate({"zscore_20": -1.5}) is False, "strict inequality"


def test_a_missing_or_null_feature_is_false_not_an_error() -> None:
    """Warm-up leaves most features null; a rule that threw there would make the first
    bars of every run unusable."""
    condition = compare("zscore_20", "<", 0.0)
    assert condition.evaluate({}) is False
    assert condition.evaluate({"zscore_20": None}) is False
    assert condition.evaluate({"other": -5.0}) is False


def test_comparing_two_features() -> None:
    condition = Compare(feature="vol_20", op=Operator.GT, other="vol_60")
    assert condition.evaluate({"vol_20": 0.02, "vol_60": 0.01}) is True
    assert condition.evaluate({"vol_20": 0.02}) is False, "missing right side is false"


def test_a_comparison_needs_exactly_one_right_hand_side() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        Compare(feature="x", op=Operator.GT)
    with pytest.raises(ValidationError, match="exactly one"):
        Compare(feature="x", op=Operator.GT, value=1.0, other="y")


@pytest.mark.parametrize(
    ("op", "left", "right", "expected"),
    [
        ("<", 1, 2, True),
        ("<=", 2, 2, True),
        (">", 3, 2, True),
        (">=", 1, 2, False),
        ("==", 2, 2, True),
        ("!=", 2, 2, False),
    ],
)
def test_every_operator(op: str, left: float, right: float, expected: bool) -> None:
    assert compare("x", op, right).evaluate({"x": left}) is expected


def test_combinators() -> None:
    row = {"a": 1.0, "b": 5.0}
    both = AllOf(conditions=(compare("a", "<", 2), compare("b", ">", 4)))
    either = AnyOf(conditions=(compare("a", ">", 99), compare("b", ">", 4)))
    assert both.evaluate(row) and either.evaluate(row)
    assert not AllOf(conditions=(compare("a", "<", 2), compare("b", "<", 4))).evaluate(row)
    assert Not(condition=compare("a", ">", 99)).evaluate(row)
    assert both.features == {"a", "b"}


def test_rules_round_trip_through_json() -> None:
    """Rules are hashed into the run id, so serialisation must be faithful."""
    rule = Rule(
        long_when=AllOf(
            conditions=(compare("vol_20_xrank", ">=", 0.99), compare("zscore_20", "<", -1.5))
        ),
        short_when=compare("zscore_20", ">", 2.0),
        weight=0.4,
    )
    assert Rule.model_validate_json(rule.model_dump_json()) == rule
    assert rule.referenced_features == {"vol_20_xrank", "zscore_20"}


def test_a_rule_needs_at_least_one_side() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        Rule(weight=0.2)


def test_targets() -> None:
    rule = Rule(long_when=compare("z", "<", -1), short_when=compare("z", ">", 1), weight=0.3)
    assert rule.target({"z": -2.0}) == 0.3
    assert rule.target({"z": 2.0}) == -0.3
    assert rule.target({"z": 0.0}) == 0.0
    assert rule.target({}) == 0.0


def test_long_wins_when_a_rule_says_both() -> None:
    """Stated precedence rather than emergent behaviour."""
    always = compare("z", ">", -999)
    rule = Rule(long_when=always, short_when=always, weight=0.5)
    assert rule.target({"z": 0.0}) == 0.5


# -- the strategy ---------------------------------------------------------------------------


class _Context:
    """Minimal DecisionContext stand-in."""

    def __init__(
        self, rows: list[dict[str, object]], positions: dict[str, float] | None = None
    ) -> None:
        self.decision_at = T0
        self.features_latest = pl.DataFrame(rows) if rows else None
        self.positions = positions or {}
        self.marks = {str(r["instrument_id"]): 100.0 for r in rows}
        self.instruments = tuple(sorted(self.marks))

    def position(self, instrument_id: str) -> float:
        return self.positions.get(instrument_id, 0.0)


def rows(*items: tuple[str, float, float]) -> list[dict[str, object]]:
    return [
        {"instrument_id": i, "zscore_20": z, "vol_20": v, "vol_20_xrank": 0.0} for i, z, v in items
    ]


def test_the_strategy_targets_every_qualifying_instrument() -> None:
    strategy = RuleStrategy(Rule(long_when=compare("zscore_20", "<", -1.5), weight=0.25))
    intents = strategy.on_decision(
        _Context(rows(("A", -2.0, 0.02), ("B", 0.0, 0.01), ("C", -3.0, 0.05)))
    )
    targets = {i.instrument_id: i.value for i in intents}
    assert targets == {"A": 0.25, "C": 0.25}


def test_max_positions_keeps_the_highest_ranked(monkeypatch: pytest.MonkeyPatch) -> None:
    """When more qualify than the cap allows, the tie-break must be deterministic."""
    strategy = RuleStrategy(
        Rule(long_when=compare("zscore_20", "<", -1.0), weight=0.5),
        max_positions=1,
        rank_by="vol_20",
    )
    context = _Context(rows(("A", -2.0, 0.02), ("B", -3.0, 0.09), ("C", -2.5, 0.05)))
    intents = strategy.on_decision(context)
    assert {i.instrument_id for i in intents} == {"B"}, "B has the highest vol_20"


def test_selection_is_stable_regardless_of_row_order() -> None:
    strategy = RuleStrategy(
        Rule(long_when=compare("zscore_20", "<", -1.0), weight=0.5),
        max_positions=2,
        rank_by="vol_20",
    )
    forward = strategy.on_decision(
        _Context(rows(("A", -2.0, 0.02), ("B", -3.0, 0.09), ("C", -2.5, 0.05)))
    )
    reverse = strategy.on_decision(
        _Context(rows(("C", -2.5, 0.05), ("B", -3.0, 0.09), ("A", -2.0, 0.02)))
    )
    assert [i.instrument_id for i in forward] == [i.instrument_id for i in reverse]
    assert {i.instrument_id for i in forward} == {"B", "C"}


def test_an_existing_position_that_no_longer_qualifies_is_closed() -> None:
    strategy = RuleStrategy(Rule(long_when=compare("zscore_20", "<", -1.5), weight=0.25))
    context = _Context(rows(("A", 0.5, 0.02)), positions={"A": 10.0})
    intents = strategy.on_decision(context)
    assert len(intents) == 1
    assert intents[0].instrument_id == "A" and intents[0].value == 0.0


def test_nothing_is_emitted_when_flat_and_nothing_qualifies() -> None:
    strategy = RuleStrategy(Rule(long_when=compare("zscore_20", "<", -1.5), weight=0.25))
    assert strategy.on_decision(_Context(rows(("A", 0.5, 0.02)))) == []


def test_an_instrument_without_a_mark_is_skipped() -> None:
    """A target weight cannot be sized without a point-in-time price."""
    strategy = RuleStrategy(Rule(long_when=compare("zscore_20", "<", -1.5), weight=0.25))
    context = _Context(rows(("A", -2.0, 0.02)))
    context.marks = {}
    assert strategy.on_decision(context) == []


def test_no_features_means_no_intents() -> None:
    strategy = RuleStrategy(Rule(long_when=compare("zscore_20", "<", -1.5), weight=0.25))
    assert strategy.on_decision(_Context([])) == []


def test_a_rule_can_be_built_from_plain_json() -> None:
    """Rules arrive from YAML/JSON with lists, not tuples; lax validation at the boundary."""
    strategy = RuleStrategy(
        {
            "long_when": {
                "kind": "all_of",
                "conditions": [
                    {"kind": "compare", "feature": "vol_20_xrank", "op": ">=", "value": 0.9},
                    {"kind": "compare", "feature": "zscore_20", "op": "<", "value": -1.5},
                ],
            },
            "weight": 0.4,
        }
    )
    assert strategy.rule.weight == 0.4
    assert strategy.rule.referenced_features == {"vol_20_xrank", "zscore_20"}


def test_max_positions_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RuleStrategy(Rule(long_when=compare("z", "<", 0)), max_positions=0)
