"""YAML-facing configuration for the application services.

DEVELOPMENT_PLAN.md sec. 6: files may be YAML for usability, but are parsed immediately
into strict Pydantic models, and it is the *resolved* configuration that is persisted and
hashed -- never the original YAML. Unknown keys are rejected, so a misspelled parameter
fails loudly instead of leaving a default economic assumption silently in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from qresearch.config import FrozenModel
from qresearch.data.adapters.base import ColumnMapping
from qresearch.data.contracts import AssetClass, BarSize, Instrument
from qresearch.data.manifests import NormalizationPolicy

if TYPE_CHECKING:
    from qresearch.application.run_backtest import BacktestConfig


class IngestConfig(FrozenModel):
    """One ingestion job: a source file, a policy, and the instruments to map."""

    source_uri: str
    source_name: str
    bar_size: BarSize
    asset_class: AssetClass
    venue: str
    calendar_id: str
    policy: NormalizationPolicy
    instruments: tuple[Instrument, ...]
    mapping: ColumnMapping = ColumnMapping()
    feed: str | None = None
    source_revision: int = 0
    normalization_version: str = "1"
    expect_complete_grid: bool = False
    """Report every gap in the regular bar grid. Correct for 24/7 crypto, not for
    equities until session calendars land."""


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML file into a plain mapping."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_backtest_config(path: Path | str) -> BacktestConfig:
    """Parse and validate a backtest config file (lax at the text boundary, see above)."""
    from qresearch.application.run_backtest import BacktestConfig

    return BacktestConfig.model_validate(load_yaml(path), strict=False)


def load_ingest_config(path: Path | str) -> IngestConfig:
    """Parse and validate an ingestion config file.

    Validation runs in lax mode here and only here. The models are strict by default so
    that in-process construction catches type slips; a YAML file is a text boundary where
    ``"crypto"`` must become ``AssetClass.CRYPTO``, ``"0.01"`` a ``Decimal``, ``PT2S`` a
    ``timedelta``, and a list a tuple. Unknown keys are still rejected -- lax coercion
    relaxes *types*, not *shape*.
    """
    return IngestConfig.model_validate(load_yaml(path), strict=False)
