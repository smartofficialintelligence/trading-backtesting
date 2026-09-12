"""Turning a form submission into a job.

The rule from ``docs/decisions.md`` D46 is enforced here and nowhere else: a launch request
is validated into the *same* :class:`~qresearch.application.run_backtest.BacktestConfig`
the YAML path produces, serialised back to YAML, and handed to the CLI. The UI never
touches the engine, and the config it writes is a file you can run yourself.

The fold preview is separate and pure — ``generate_folds`` executes nothing — so a plan can
be checked before spending the time on it.
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import Field, ValidationError

from qresearch.application.run_backtest import BacktestConfig
from qresearch.config import FrozenModel
from qresearch.data.catalog import DatasetCatalog
from qresearch.features.registry import build_feature, build_transform
from qresearch.research.splits import TimeRange
from qresearch.research.walk_forward import FixedSplitPlan, generate_folds
from qresearch.strategy.registry import build_strategy
from qresearch.time import UtcDatetime


class LaunchError(ValueError):
    """The submitted configuration is not valid. Carries the field-level detail."""

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class DatasetSummary(FrozenModel):
    """What a dataset picker needs."""

    dataset_id: str
    asset_class: str
    venue: str
    bar_size: str
    calendar_id: str
    instrument_ids: tuple[str, ...]
    range_start: UtcDatetime
    range_end: UtcDatetime
    row_count: int
    warning_count: int
    error_count: int


def dataset_summaries(catalog: DatasetCatalog) -> list[DatasetSummary]:
    out = []
    for dataset_id in catalog.list_datasets():
        manifest = catalog.resolve(dataset_id)
        identity = manifest.identity
        out.append(
            DatasetSummary(
                dataset_id=str(dataset_id),
                asset_class=identity.asset_class.value,
                venue=identity.venue,
                bar_size=identity.bar_size,
                calendar_id=identity.calendar_id,
                instrument_ids=tuple(str(i) for i in identity.instrument_ids),
                range_start=identity.range_start,
                range_end=identity.range_end,
                row_count=manifest.row_count,
                warning_count=len(manifest.validation.warnings),
                error_count=len(manifest.validation.errors),
            )
        )
    return out


class FoldSummary(FrozenModel):
    fold: int
    role: str
    start: UtcDatetime
    end: UtcDatetime
    duration_s: float


class PreviewResult(FrozenModel):
    """What a plan would do, without doing it."""

    folds: tuple[FoldSummary, ...] = ()
    fold_count: int = 0
    span: TimeRange | None = None
    error: str | None = None


def preview_plan(config: dict[str, Any], catalog: DatasetCatalog) -> PreviewResult:
    """Expand a plan into folds without running anything.

    Catching a purge that swallows the data, or a plan that fits no folds at all, costs a
    second here rather than minutes after launching.
    """
    try:
        parsed = _parse(config)
    except LaunchError as error:
        return PreviewResult(error=str(error))

    manifest = catalog.resolve(parsed.dataset_id)
    span = parsed.span or TimeRange(
        start=manifest.identity.range_start, end=manifest.identity.range_end
    )
    try:
        folds = (
            [parsed.plan.fold()]
            if isinstance(parsed.plan, FixedSplitPlan)
            else generate_folds(parsed.plan, span)
        )
    except ValueError as error:
        return PreviewResult(span=span, error=str(error))

    summaries = [
        FoldSummary(
            fold=fold.index,
            role=split.role.value,
            start=split.range.start,
            end=split.range.end,
            duration_s=split.range.duration.total_seconds(),
        )
        for fold in folds
        for split in fold.splits()
    ]
    return PreviewResult(folds=tuple(summaries), fold_count=len(folds), span=span)


class LaunchRequest(FrozenModel):
    """A form submission. Everything else is the config itself."""

    config: dict[str, Any]
    label: str | None = None
    root: str = "data"
    runs: str = "runs"


def _parse(config: dict[str, Any]) -> BacktestConfig:
    """Validate through the same path the YAML loader uses, then build the components.

    Pydantic checks the shape but not the vocabulary: ``{"kind": "nope"}`` is a perfectly
    valid ``StrategyRef``, and the registry lookup would not fail until the run was
    minutes underway. Constructing each component here turns an unknown kind, an unknown
    parameter, or an out-of-range value into an error before anything is queued.
    """
    try:
        parsed = BacktestConfig.model_validate(config, strict=False)
    except ValidationError as error:
        raise LaunchError(
            f"{error.error_count()} invalid field(s)",
            [
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                for e in error.errors()
            ],
        ) from error

    problems: list[dict[str, Any]] = []
    for index, feature in enumerate(parsed.features):
        problems.extend(_check(f"features.{index}", build_feature, feature.kind, feature.params))
    for index, transform in enumerate(parsed.transforms):
        problems.extend(
            _check(
                f"transforms.{index}",
                build_transform,
                transform.kind,
                transform.columns,
                transform.params,
            )
        )
    problems.extend(
        _check("strategy", build_strategy, parsed.strategy.kind, parsed.strategy.params)
    )
    if problems:
        raise LaunchError(f"{len(problems)} component(s) could not be built", problems)
    return parsed


def _check(field: str, build: Any, *args: Any) -> list[dict[str, Any]]:
    try:
        build(*args)
    except (KeyError, ValueError, TypeError) as error:
        return [{"field": field, "message": str(error).strip("'")}]
    return []


def config_to_yaml(config: dict[str, Any]) -> str:
    """Validate and render the config a job will run.

    Round-trips through ``BacktestConfig`` so the YAML written is exactly what the CLI
    would parse — a form cannot express something the file format cannot.
    """
    parsed = _parse(config)
    payload = parsed.model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)


def estimate(config: dict[str, Any], catalog: DatasetCatalog) -> dict[str, Any]:
    """Rough shape of the work a launch would create, for the confirm step."""
    parsed = _parse(config)
    preview = preview_plan(config, catalog)
    runs = len(parsed.cost_scenarios) * len(parsed.fill_rules)
    return {
        "runs": runs,
        "folds_per_run": preview.fold_count,
        "simulations": runs * preview.fold_count * (2 if parsed.evaluate_validation else 1),
        "cost_scenarios": list(parsed.cost_scenarios),
        "fill_rules": [r.value for r in parsed.fill_rules],
    }


DEFAULT_CONFIG: dict[str, Any] = {
    "features": [{"kind": "lagged_return", "params": {"lag": 1}}],
    "transforms": [],
    "strategy": {"kind": "lagged_signal", "params": {"feature": "ret_1", "weight": 0.4}},
    "simulation": {"initial_cash": 10000.0},
    # ISO-8601 durations, not timedelta objects: these are rendered straight into form
    # fields and posted back, so they must already be in the wire format the config
    # parser accepts. A timedelta would serialise as "6:00:00", which does not parse.
    "plan": {
        "kind": "rolling",
        "train": "PT6H",
        "validation": "PT2H",
        "test": "PT2H",
        "purge": "PT2M",
        "label_horizon": "PT1M",
        "warmup": "PT1H",
    },
    "cost_scenarios": ["base"],
    "fill_rules": ["open_of_current_bar", "next_open_after_eligibility"],
}
"""A sensible starting point for the form: one feature, a signal strategy, both fill
rules so the bracket is the default rather than something to remember."""


class LaunchOutcome(FrozenModel):
    job_id: str
    config_yaml: str
    command: tuple[str, ...]
    estimate: dict[str, Any] = Field(default_factory=dict)
