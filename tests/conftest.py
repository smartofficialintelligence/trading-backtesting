"""Shared fixtures: a synthetic source file and an ingested dataset."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import pytest

from qresearch.application.ingest import IngestOutcome, ingest_bars
from qresearch.data.adapters.base import ColumnMapping, IngestRequest
from qresearch.data.adapters.csv_parquet import LocalFileBarAdapter
from qresearch.data.catalog import DatasetCatalog
from qresearch.data.contracts import AssetClass, PriceAdjustment
from qresearch.data.manifests import (
    DuplicatePolicy,
    NormalizationPolicy,
    TimestampLabel,
)
from qresearch.data.synthetic import (
    SOURCE_NAME,
    SyntheticSpec,
    instruments_for,
    write_source_csv,
)

NORMALIZATION_VERSION = "test-1"


@pytest.fixture
def spec() -> SyntheticSpec:
    return SyntheticSpec()


@pytest.fixture
def policy() -> NormalizationPolicy:
    return NormalizationPolicy(
        timestamp_label=TimestampLabel.BAR_START,
        publication_latency=_dt.timedelta(seconds=2),
        price_adjustment=PriceAdjustment.NOT_APPLICABLE,
        duplicates=DuplicatePolicy.KEEP_HIGHEST_REVISION,
        volume_unit="base_asset",
    )


@pytest.fixture
def source_csv(tmp_path: Path, spec: SyntheticSpec) -> Path:
    return write_source_csv(tmp_path / "source" / "bars.csv", spec)


@pytest.fixture
def catalog(tmp_path: Path) -> DatasetCatalog:
    return DatasetCatalog(tmp_path / "data")


def make_request(
    uri: Path, spec: SyntheticSpec, policy: NormalizationPolicy, **overrides: object
) -> IngestRequest:
    """Build an ingest request for the synthetic source, with optional overrides."""
    base: dict[str, object] = {
        "uri": str(uri),
        "bar_size": "1m",
        "policy": policy,
        "instruments": instruments_for(spec),
        "source_name": SOURCE_NAME,
        "mapping": ColumnMapping(
            trade_count="trades", available_at="available_at", revision="revision"
        ),
    }
    return IngestRequest.model_validate(base | overrides)


def run_ingest(
    catalog: DatasetCatalog, request: IngestRequest, **overrides: object
) -> IngestOutcome:
    base: dict[str, object] = {
        "catalog": catalog,
        "adapter": LocalFileBarAdapter(),
        "asset_class": AssetClass.CRYPTO,
        "venue": "SYNTH",
        "calendar_id": "24x7:1",
        "normalization_version": NORMALIZATION_VERSION,
        "created_by": "pytest",
        "expect_complete_grid": True,
    }
    return ingest_bars(request, **(base | overrides))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestedFixture:
    catalog: DatasetCatalog
    outcome: IngestOutcome
    spec: SyntheticSpec
    policy: NormalizationPolicy
    source: Path

    @property
    def dataset_id(self) -> str:
        return self.outcome.manifest.dataset_id


@pytest.fixture
def ingested(
    catalog: DatasetCatalog, source_csv: Path, spec: SyntheticSpec, policy: NormalizationPolicy
) -> IngestedFixture:
    outcome = run_ingest(catalog, make_request(source_csv, spec, policy))
    return IngestedFixture(
        catalog=catalog, outcome=outcome, spec=spec, policy=policy, source=source_csv
    )
