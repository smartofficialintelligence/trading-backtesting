"""Rough local benchmark: ingest, features, and a walk-forward run at modest scale.

Records wall-clock and peak RSS to a JSON file so later optimisation work has a
baseline (DEVELOPMENT_PLAN.md sec. 9: measure before optimising). Not a test.

    uv run python scripts/benchmark.py --instruments 5 --days 5 --out benchmarks/latest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import resource
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from qresearch.application.ingest import ingest_bars
from qresearch.application.run_backtest import BacktestConfig, run_backtest
from qresearch.artifacts.contracts import FeatureRef, StrategyRef
from qresearch.artifacts.local import LocalArtifactStore
from qresearch.data.adapters.base import ColumnMapping, IngestRequest
from qresearch.data.adapters.csv_parquet import LocalFileBarAdapter
from qresearch.data.catalog import DatasetCatalog
from qresearch.data.contracts import AssetClass, PriceAdjustment
from qresearch.data.manifests import DuplicatePolicy, NormalizationPolicy, TimestampLabel
from qresearch.data.synthetic import SOURCE_NAME, SyntheticAsset, SyntheticSpec, instruments_for, write_source_csv
from qresearch.research.walk_forward import WalkForwardPlan
from qresearch.simulation.engine import SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", type=int, default=5)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/latest.json"))
    args = parser.parse_args()

    assets = tuple(
        SyntheticAsset(f"A{i}-USD", f"CRYPTO:A{i}USD", 100.0 * (i + 1), 0.001, 50.0)
        for i in range(args.instruments)
    )
    spec = replace(SyntheticSpec(), assets=assets, minutes=args.days * 1440, gap_minutes=(), delayed_minutes=(), duplicate_minutes=())
    timings: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        t = time.perf_counter()
        source = write_source_csv(root / "src.csv", spec)
        timings["generate_s"] = time.perf_counter() - t

        request = IngestRequest(
            uri=str(source), bar_size="1m",
            policy=NormalizationPolicy(
                timestamp_label=TimestampLabel.BAR_START, publication_latency=dt.timedelta(seconds=2),
                price_adjustment=PriceAdjustment.NOT_APPLICABLE, duplicates=DuplicatePolicy.KEEP_HIGHEST_REVISION,
                volume_unit="base_asset",
            ),
            instruments=instruments_for(spec), source_name=SOURCE_NAME,
            mapping=ColumnMapping(trade_count="trades", available_at="available_at", revision="revision"),
        )
        catalog = DatasetCatalog(root / "data")
        t = time.perf_counter()
        outcome = ingest_bars(
            request, catalog=catalog, adapter=LocalFileBarAdapter(), asset_class=AssetClass.CRYPTO,
            venue="SYNTH", calendar_id="24x7:1", normalization_version="1", created_by="benchmark",
        )
        timings["ingest_s"] = time.perf_counter() - t

        config = BacktestConfig(
            dataset_id=outcome.manifest.dataset_id,
            features=(FeatureRef(kind="lagged_return", params={"lag": 1}), FeatureRef(kind="rolling_volatility", params={"window": 20})),
            strategy=StrategyRef(kind="lagged_signal", params={"feature": "ret_1", "weight": 0.2}),
            simulation=SimulationConfig(),
            plan=WalkForwardPlan(train=dt.timedelta(days=1), test=dt.timedelta(hours=12), purge=dt.timedelta(minutes=5)),
            cost_scenarios=("base",),
        )
        t = time.perf_counter()
        result = run_backtest(config, catalog=catalog, store=LocalArtifactStore(root / "runs"))
        timings["backtest_s"] = time.perf_counter() - t
        folds = len({f.fold for f in result.folds})

    record = {
        "captured_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "instruments": args.instruments,
        "days": args.days,
        "bars": outcome.manifest.row_count,
        "folds": folds,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        **{k: round(v, 3) for k, v in timings.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
