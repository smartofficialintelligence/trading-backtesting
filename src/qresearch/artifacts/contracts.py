"""Run specifications and results.

A :class:`RunSpec` is the *resolved* configuration of one backtest: dataset version,
feature fingerprints, strategy and parameters, simulation and cost settings, fold plan,
seed, and code revision. Its canonical hash is the ``run_id``, so an identical experiment
is recognisable as such and a different one never collides (ARCHITECTURE.md sec. 6).

The human label is excluded from the hash: renaming a run does not make it a new run.
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from qresearch.config import FrozenModel
from qresearch.ids import DatasetId, RunId, content_hash
from qresearch.research.metrics import Metrics
from qresearch.research.splits import SplitRole, TimeRange
from qresearch.research.walk_forward import FixedSplitPlan, WalkForwardPlan
from qresearch.simulation.engine import SimulationConfig
from qresearch.simulation.events import WarningRecord
from qresearch.time import UtcDatetime

ARTIFACT_SCHEMA_VERSION = 1


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class FeatureRef(FrozenModel):
    kind: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class CrossSectionalRef(FrozenModel):
    """A rank or mean computed across instruments at each instant.

    Separate from ``FeatureRef`` because the timing is different in kind: a per-instrument
    feature is usable when its own inputs are in, a cross-sectional value only when every
    member's is (ARCHITECTURE.md sec. 9, cross-sectional asynchrony).
    """

    column: str = Field(min_length=1)
    min_members: int = Field(default=2, ge=2)
    pct: bool = True
    descending: bool = False
    statistic: Literal["rank", "mean"] = "rank"
    """``rank`` produces ``{column}_xrank``; ``mean`` produces ``{column}_xmean``, the
    equal-weight average across instruments."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.statistic == "mean" and (not self.pct or self.descending):
            raise ValueError("pct and descending apply to ranks, not to a cross-sectional mean")
        return self

    @property
    def name(self) -> str:
        return f"{self.column}_x{self.statistic}"


class TransformRef(FrozenModel):
    kind: str = Field(min_length=1)
    columns: tuple[str, ...] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class StrategyRef(FrozenModel):
    kind: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class RunSpec(FrozenModel):
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    dataset_id: DatasetId
    instrument_ids: tuple[str, ...] = Field(min_length=1)
    bar_size: str
    calendar_id: str
    features: tuple[FeatureRef, ...] = ()
    cross_sectional: tuple[CrossSectionalRef, ...] = ()
    """Ranks computed across instruments after the per-instrument features."""

    feature_fingerprints: tuple[str, ...] = ()
    """Resolved ``FeatureSpec.fingerprint`` per feature, so a code change to a feature
    implementation (which bumps its version) changes the run id."""

    transforms: tuple[TransformRef, ...] = ()
    strategy: StrategyRef
    simulation: SimulationConfig
    plan: WalkForwardPlan | FixedSplitPlan
    span: TimeRange
    cost_scenario: str = "base"
    seed: int = 0
    evaluate_validation: bool = True
    code_revision: str | None = None
    """``<git sha>`` or ``<git sha>-dirty-<diff hash>``; None outside a repository."""

    experiment_id: str | None = None
    label: str | None = None

    excluded: tuple[TimeRange, ...] = ()
    """Ranges no fold may touch; folds that would are skipped. Resolved from the sealed
    test windows when a config sets ``exclude_sealed`` (``qresearch.research.holdout``)."""

    unsealed: tuple[str, ...] = ()
    """Sealed windows this run was allowed to evaluate. Non-empty means it spent a holdout."""

    def identity(self) -> dict[str, Any]:
        """Everything that makes this an exact run, code revision included."""
        data = self.canonical_dict()
        data.pop("label", None)
        # Added after runs had been recorded: left out of the hash while unused, so every
        # earlier run id still resolves to the same run.
        for key in ("excluded", "unsealed"):
            if not data.get(key):
                data.pop(key, None)
        for ref in data.get("cross_sectional", ()):
            if ref.get("statistic") == "rank":
                ref.pop("statistic")
        return data

    def config_identity(self) -> dict[str, Any]:
        """Everything that makes this the same *experiment configuration*.

        Drops the code revision as well as the label, so editing a docstring -- or any
        other change that does not alter the configuration -- leaves runs groupable.
        """
        data = self.identity()
        data.pop("code_revision", None)
        return data

    @property
    def run_id(self) -> RunId:
        """Exact provenance: this configuration, produced by this code."""
        return RunId(f"run_{content_hash(self.identity())}")

    @property
    def config_id(self) -> str:
        """Stable across code revisions: groups every run of one configuration."""
        return f"cfg_{content_hash(self.config_identity())}"


class FoldMetrics(FrozenModel):
    fold: int = Field(ge=0)
    role: SplitRole
    range: TimeRange
    decisions: int = Field(ge=0)
    metrics: Metrics


class RunResult(FrozenModel):
    run_id: RunId
    status: RunStatus
    started_at: UtcDatetime
    finished_at: UtcDatetime | None = None
    error: str | None = None
    folds: tuple[FoldMetrics, ...] = ()
    aggregate: dict[str, Metrics] = Field(default_factory=dict)
    """By split role ("validation", "test"): metrics over the stitched fold curves."""

    warnings: tuple[WarningRecord, ...] = ()
    economic_digest: str | None = None
    """Hash of canonical fills and equity curve, for reproducibility comparison across
    runs of the same spec (DEVELOPMENT_PLAN.md sec. 7)."""

    artifact_files: tuple[str, ...] = ()

    @property
    def duration(self) -> _dt.timedelta | None:
        return None if self.finished_at is None else self.finished_at - self.started_at
