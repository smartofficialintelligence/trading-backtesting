"""A strategy assembled from conditions over features.

Deliberately not a language. A rule is a tree of comparisons combined with all/any/not,
producing one target weight. No loops, no arithmetic, no state. Anything it cannot express
is a ``Strategy`` class, and that path already works -- which is the guard against this
growing into a DSL nobody can test.

Rules are *data*: serialisable, part of the ``RunSpec``, and hashed into the ``run_id``, so
a rule change is a new run and the rule is stored alongside its results. Being data rather
than code also means there is no sandbox question.

Evaluation is per instrument, per decision, against the feature row available at that
instant. A comparison whose feature is missing or null is **false**, never an error: during
warm-up most features are null, and a rule that threw there would make the first bars of
every run unusable. The cost is that a typo'd feature name reads as "never true" rather
than failing loudly -- which is why :meth:`Rule.referenced_features` exists and the
launcher checks it against the configured feature set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.simulation.events import OrderIntent
from qresearch.strategy.contracts import DecisionContext


class Operator(StrEnum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="

    def compare(self, left: float, right: float) -> bool:
        match self:
            case Operator.LT:
                return left < right
            case Operator.LE:
                return left <= right
            case Operator.GT:
                return left > right
            case Operator.GE:
                return left >= right
            case Operator.EQ:
                return left == right
            case Operator.NE:
                return left != right


class Compare(FrozenModel):
    """``feature <op> value``, where value is a constant or another feature."""

    kind: Literal["compare"] = "compare"
    feature: str = Field(min_length=1)
    op: Operator
    value: float | None = None
    other: str | None = None
    """Compare against another feature instead of a constant."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.value is None) == (self.other is None):
            raise ValueError(
                f"comparison on {self.feature!r} needs exactly one of 'value' or 'other'"
            )
        return self

    def evaluate(self, row: Mapping[str, Any]) -> bool:
        left = row.get(self.feature)
        if left is None:
            return False
        right: Any = self.value
        if self.other is not None:
            right = row.get(self.other)
            if right is None:
                return False
        try:
            return self.op.compare(float(left), float(right))
        except (TypeError, ValueError):
            return False

    @property
    def features(self) -> set[str]:
        return {self.feature} | ({self.other} if self.other else set())


class AllOf(FrozenModel):
    kind: Literal["all_of"] = "all_of"
    conditions: tuple[Condition, ...] = Field(min_length=1)

    def evaluate(self, row: Mapping[str, Any]) -> bool:
        return all(c.evaluate(row) for c in self.conditions)

    @property
    def features(self) -> set[str]:
        return set().union(*(c.features for c in self.conditions))


class AnyOf(FrozenModel):
    kind: Literal["any_of"] = "any_of"
    conditions: tuple[Condition, ...] = Field(min_length=1)

    def evaluate(self, row: Mapping[str, Any]) -> bool:
        return any(c.evaluate(row) for c in self.conditions)

    @property
    def features(self) -> set[str]:
        return set().union(*(c.features for c in self.conditions))


class Not(FrozenModel):
    kind: Literal["not"] = "not"
    condition: Condition

    def evaluate(self, row: Mapping[str, Any]) -> bool:
        return not self.condition.evaluate(row)

    @property
    def features(self) -> set[str]:
        return self.condition.features


Condition = Annotated[Compare | AllOf | AnyOf | Not, Field(discriminator="kind")]

AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


class Rule(FrozenModel):
    """Enter long when ``long_when`` holds, short when ``short_when`` holds, else flat."""

    long_when: Condition | None = None
    short_when: Condition | None = None
    weight: float = Field(default=0.2, gt=0)
    """Target weight per position, as a fraction of equity."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.long_when is None and self.short_when is None:
            raise ValueError("a rule needs at least one of long_when or short_when")
        return self

    def target(self, row: Mapping[str, Any]) -> float:
        """Target weight for one instrument at one instant.

        Long is checked first: if a rule somehow says both, the strategy takes the long
        rather than netting to flat, and that precedence is stated rather than emergent.
        """
        if self.long_when is not None and self.long_when.evaluate(row):
            return self.weight
        if self.short_when is not None and self.short_when.evaluate(row):
            return -self.weight
        return 0.0

    @property
    def referenced_features(self) -> set[str]:
        out: set[str] = set()
        for condition in (self.long_when, self.short_when):
            if condition is not None:
                out |= condition.features
        return out


class RuleStrategy:
    """Applies a :class:`Rule` to every instrument with a published feature row.

    ``max_positions`` caps how many names are held at once. When more qualify than the cap
    allows, the tie is broken by ``rank_by`` (largest first, or smallest with
    ``rank_ascending``) and then by instrument id, so the choice is deterministic rather
    than dependent on dictionary order.
    """

    strategy_id = "rule:v1"

    def __init__(
        self,
        rule: Rule | Mapping[str, Any],
        *,
        max_positions: int | None = None,
        rank_by: str | None = None,
        rank_ascending: bool = False,
    ) -> None:
        # Lax at the boundary, like the YAML config loader (D5): a rule arriving from
        # JSON or YAML has lists where the model declares tuples, and strings where it
        # declares enums. Shape is still enforced -- unknown keys are rejected.
        self.rule = rule if isinstance(rule, Rule) else Rule.model_validate(rule, strict=False)
        if max_positions is not None and max_positions < 1:
            raise ValueError(f"max_positions must be at least 1, got {max_positions}")
        self.max_positions = max_positions
        self.rank_by = rank_by
        self.rank_ascending = rank_ascending

    @property
    def required_features(self) -> set[str]:
        """Feature columns this strategy reads.

        The orchestrator checks these against the computed feature set. Without it, a
        renamed or mistyped feature makes every comparison false (see the module
        docstring) and the run completes with zero trades and no error -- which looks
        exactly like a strategy that found no opportunities.
        """
        needed = set(self.rule.referenced_features)
        if self.rank_by:
            needed.add(self.rank_by)
        return needed

    def reset(self) -> None:
        return None

    def on_decision(self, context: DecisionContext) -> Sequence[OrderIntent]:
        rows = self._rows(context)
        wanted: list[tuple[str, float]] = []
        for instrument_id, row in sorted(rows.items()):
            if instrument_id not in context.marks:
                continue  # cannot size a target weight without a point-in-time mark
            target = self.rule.target(row)
            if target != 0.0:
                wanted.append((instrument_id, target))

        if self.max_positions is not None and len(wanted) > self.max_positions:
            wanted = self._select(wanted, rows)

        targets = dict(wanted)
        intents = []
        for instrument_id in sorted(set(targets) | set(context.positions)):
            if instrument_id not in context.marks:
                continue
            target = targets.get(instrument_id, 0.0)
            held = context.position(instrument_id)
            if target == 0.0 and held == 0.0:
                continue
            intents.append(
                OrderIntent.target_weight(instrument_id, target, at=context.decision_at, tag="rule")
            )
        return intents

    def _select(
        self, wanted: list[tuple[str, float]], rows: Mapping[str, Mapping[str, Any]]
    ) -> list[tuple[str, float]]:
        def key(item: tuple[str, float]) -> tuple[float, str]:
            value = rows[item[0]].get(self.rank_by) if self.rank_by else None
            score = float(value) if isinstance(value, int | float) else float("-inf")
            return (score if not self.rank_ascending else -score, item[0])

        ranked = sorted(wanted, key=key, reverse=True)
        assert self.max_positions is not None
        return ranked[: self.max_positions]

    @staticmethod
    def _rows(context: DecisionContext) -> dict[str, dict[str, Any]]:
        frame = context.features_latest
        if frame is None or frame.is_empty():
            return {}
        return {str(row["instrument_id"]): row for row in frame.iter_rows(named=True)}
