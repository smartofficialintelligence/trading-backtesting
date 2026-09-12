"""Run a parameter grid and compare the results on validation.

A stopgap until Stage 9's sweep runner: it expands a grid, runs each point under one
``experiment_id``, and prints a table. Two habits it enforces, because a sweep is the
easiest way to fool yourself:

* **Comparison is on validation, not test.** Test is reported alongside so you can see it,
  but choosing on it is how a search result becomes a fantasy.
* **The trial count is printed with the winner.** The best of twelve is not the same claim
  as a result obtained once (``docs/leakage_checklist.md``, multiple comparisons).

    uv run python scripts/sweep.py --base configs/examples/mean_reversion_most_volatile.yaml
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from qresearch.application.config import load_backtest_config
from qresearch.application.run_backtest import BacktestConfig, run_backtest
from qresearch.artifacts.local import LocalArtifactStore
from qresearch.data.catalog import DatasetCatalog
from qresearch.simulation.execution import FillRule


def set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``a.b.c`` on a nested mapping, creating nothing that does not exist."""
    node: Any = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


LINKED = "linked"
"""Grid key prefix whose values are mappings of dotted path -> value, applied together.

Needed for parameters that must move in step. A symmetric entry threshold is the obvious
case: varying the long and short sides independently would generate mismatched pairs
(long at -2, short at +3) that nobody intended to test.
"""


def expand(
    base: dict[str, Any], grid: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Cartesian product of the grid, applied to copies of the base config."""
    keys = sorted(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        config = copy.deepcopy(base)
        label: dict[str, Any] = {}
        for dotted, value in zip(keys, values, strict=True):
            if dotted.startswith(LINKED):
                if not isinstance(value, dict):
                    raise TypeError(f"{dotted!r} values must be mappings of path -> value")
                for inner, inner_value in value.items():
                    set_path(config, inner, inner_value)
                label.update({k.split(".")[-1]: v for k, v in value.items()})
            else:
                set_path(config, dotted, value)
                label[dotted.split(".")[-1]] = value
        out.append((label, config))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--grid", type=Path, help="JSON mapping of dotted path -> list of values")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--experiment", default="sweep")
    args = parser.parse_args()

    base = yaml.safe_load(args.base.read_text())
    grid = json.loads(args.grid.read_text()) if args.grid else {}
    points = expand(base, grid)
    print(f"{len(points)} trial(s)\n")

    catalog, store = DatasetCatalog(args.root), LocalArtifactStore(args.runs)
    rows = []
    for index, (point, config) in enumerate(points, start=1):
        config["experiment_id"] = args.experiment
        config["label"] = " ".join(f"{k}={v}" for k, v in sorted(point.items()))
        parsed = BacktestConfig.model_validate(config, strict=False)
        print(f"[{index}/{len(points)}] {config['label']}", flush=True)
        result = run_backtest(
            parsed,
            catalog=catalog,
            store=store,
            cost_scenario="base",
            fill_rule=FillRule.OPEN_OF_CURRENT_BAR,
        )
        row: dict[str, Any] = {"trial": config["label"], "run_id": result.run_id}
        for role in ("validation", "test"):
            metrics = result.aggregate.get(role)
            row[f"{role}_return"] = metrics.total_return if metrics else None
            row[f"{role}_sharpe"] = metrics.sharpe if metrics else None
            row[f"{role}_trades"] = metrics.trades.trade_count if metrics and metrics.trades else 0
            row[f"{role}_costs"] = metrics.cost_fraction if metrics else None
        rows.append(row)

    table = pl.DataFrame(rows).sort("validation_return", descending=True)
    with pl.Config(tbl_rows=100, tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=44):
        print("\n" + str(table))
    best = table.row(0, named=True)
    print(
        f"\nBest on VALIDATION: {best['trial']}  "
        f"({best['validation_return']:+.3%}, test {best['test_return']:+.3%})"
    )
    print(
        f"This is the best of {len(points)} trials. Report that count with the number: "
        "the maximum of many draws is biased upward, and five days is one regime."
    )
    # Round-trip the loader so the printed configs are the ones the CLI would run.
    assert load_backtest_config(args.base).dataset_id


if __name__ == "__main__":
    main()
