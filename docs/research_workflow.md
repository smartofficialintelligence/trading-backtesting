# Research workflow

The intended loop, and where each guardrail sits.

## 1. State the hypothesis

Before any run: universe, bar size, horizon, primary metric, and cost assumption. Put
the primary metric and the `experiment_id` in the config. Everything under that id is
one experiment; every trial is a stored run.

## 2. Ingest once, immutably

```sh
qresearch data ingest -c configs/my_source.yaml --root data
qresearch data inspect <dataset-id> --root data      # read the findings
qresearch data validate <dataset-id> --root data     # re-check with the calendar
```

The dataset id is the identity of the data you tested on. Runs reference it; corrections
are new ids.

## 3. Write features against the leakage checkers

```python
from qresearch.features.leakage import assert_prefix_invariant, assert_future_insensitive
from qresearch.features.pipeline import compute_features

compute = lambda bars: compute_features(bars, [MyFeature()])
assert_prefix_invariant(compute, fixture_with_gaps_and_late_bars)
assert_future_insensitive(compute, fixture_with_gaps_and_late_bars)
```

A feature is one Polars expression plus a `FeatureSpec` declaring its lookback. The
pipeline derives availability; you never write it. Register the feature in
`features/registry.py` to use it from YAML.

## 4. Reserve the test period; tune on validation

The plan's `validation` range is where parameters are chosen; `test` is evaluated
alongside but must not drive selection. Both are stored with the run, separately
labelled. `runs compare ... --role validation` for tuning, `--role test` once.

## 5. Run with sensitivity

```sh
qresearch backtest run -c configs/examples/crypto_momentum.yaml --root data --runs runs
```

One run per cost scenario. Read `base` first, then check `stressed` still has the sign
you expect. Read the warnings block before the Sharpe.

## 6. Compare and record

```sh
qresearch runs compare <id> <id> --role test
qresearch runs index --runs runs          # DuckDB: runs, run_folds
```

Assumptions (dataset, cost scenario, fill rule, warnings) print before performance so
they are read first.

## 7. Reproduce before promoting

```sh
qresearch runs reproduce <id> --root data --runs runs
```

Re-executes from the stored spec and compares the economic digest. A promoted result
should reproduce exactly.

## Interpreting metrics

All metrics are per split role over the stitched fold curves; per-fold values are in
`metrics.json`. Annualisation is labelled on every record (`24x7` and `XNYS` differ).
`period_hit_rate` is a period-level proxy, not trade-level. `cost_fraction` is total
friction over starting equity — compare it to `total_return`.
