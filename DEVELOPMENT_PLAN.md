# Development Plan

## 1. Delivery approach

Build the smallest end-to-end research loop first, then deepen realism and scale. Each stage must leave the repository runnable, tested, and documented. Later stages should not weaken earlier timing or reproducibility invariants.

The MVP is complete when a researcher can ingest a small bar dataset, define causal features and a multi-asset strategy, run deterministic train/validation/test and walk-forward backtests with conservative costs, and compare persisted experiments from the CLI.

## 2. Proposed repository structure

```text
trading-backtesting/
  pyproject.toml
  uv.lock                         # once dependencies are resolved
  README.md
  ARCHITECTURE.md
  DEVELOPMENT_PLAN.md
  LICENSE
  .gitignore
  configs/
    examples/
      crypto_momentum.yaml
      equities_mean_reversion.yaml
  src/
    qresearch/
      __init__.py
      cli.py
      config.py
      time.py
      ids.py
      data/
        contracts.py
        catalog.py
        manifests.py
        validation.py
        point_in_time.py
        calendars.py
        adapters/
          base.py
          csv_parquet.py
      features/
        contracts.py
        pipeline.py
        technical.py
        transforms.py
      strategy/
        contracts.py
        examples.py
      simulation/
        clock.py
        events.py
        engine.py
        orders.py
        execution.py
        costs.py
        portfolio.py
        constraints.py
      research/
        splits.py
        walk_forward.py
        metrics.py
        experiments.py
      artifacts/
        contracts.py
        local.py
        schemas.py
      application/
        ingest.py
        run_backtest.py
        compare_runs.py
  tests/
    unit/
    integration/
    contracts/
    golden/
      data/
  docs/
    data_contracts.md
    timestamp_semantics.md
    execution_assumptions.md
  scripts/                         # maintenance only; no business logic
```

Keep modules together until they become genuinely large. In particular, do not create one file per domain class, a generic repository pattern, a dependency-injection framework, or abstract base classes that have only one foreseeable implementation.

## 3. MVP boundaries

### Included

- Python 3.12+, Polars, DuckDB, PyArrow/Parquet, Pydantic 2, Typer (or standard-library argparse), pytest, and a locked dependency workflow.
- Canonical 1-minute bars and deterministic 5-minute derivation.
- Crypto 24/7 and one US equity exchange-calendar path.
- Local provider-agnostic CSV/Parquet import; one real data-provider adapter can follow once provider/licensing is selected.
- Multiple assets in one portfolio and one quote currency per run.
- Causal rolling/lagged features and a small fitted-transform example.
- Market orders filled at the first eligible subsequent bar open.
- Explicit fixed/estimated spread, configurable slippage, commissions/fees, latency, and participation constraints based only on liquidity estimates available by order submission.
- Long and optionally short positions; cash, positions, realized/unrealized P&L, and exposure.
- Fixed temporal splits and rolling/expanding walk-forward folds with purge support.
- Run artifacts and DuckDB-backed comparison from a CLI.

### Excluded

- Live trading, broker APIs, streaming infrastructure, UI, distributed compute, options, L2 books, news, funding, borrow inventory, tax accounting, and optimizer farms.
- Claims of limit/stop execution realism using only bars.
- Cross-currency portfolios until point-in-time FX and cash conversion are implemented.

## 4. Staged implementation

### Stage 0 — Project foundation

**Goal:** A reproducible, typed, testable Python package with configuration and quality gates.

Deliverables:

- Initialize Git and a `src/` package using Python 3.12+.
- Add `pyproject.toml`, lockfile, Ruff, a type checker, pytest, coverage, and pre-commit configuration.
- Define Pydantic base configuration with strict timezone-aware UTC validation.
- Define typed IDs and canonical serialization/hash utilities.
- Add structured logging with `run_id` context.
- Add CI for lint, types, unit tests, and package build.
- Record architecture decisions that alter the contracts in this document.

Acceptance criteria:

- A clean checkout installs from the lockfile and passes all quality checks.
- Naive/non-UTC boundary timestamps are rejected or explicitly normalized according to one documented policy.
- Canonical configuration hashes are stable across mapping order and machines.

### Stage 1 — Data foundation

**Goal:** Convert source bars into a validated, immutable, queryable point-in-time dataset.

Deliverables:

- Implement `Instrument`, `Bar`, `DatasetManifest`, and validation reports.
- Implement a local CSV/Parquet adapter with explicit source timestamp semantics.
- Normalize data to partitioned Parquet and create content-addressed manifests.
- Build DuckDB views/catalog from manifests.
- Add duplicate, gap, OHLC, session, timestamp, and monotonicity checks.
- Implement exchange calendar handling and 1-minute to 5-minute resampling.
- Implement point-in-time scans filtered by `available_at`.
- Add a tiny synthetic dataset containing gaps, delayed observations, duplicate source rows, and multiple assets.

Acceptance criteria:

- Re-ingesting the same source and policy yields the same logical dataset ID.
- Corrections produce a new manifest without changing old data.
- Five-minute bars never appear before every included minute is available.
- Queries at time `t` return no row with `available_at > t`.
- Provider timestamp behavior is captured in contract tests.

### Stage 2 — Features and leakage defenses

**Goal:** Produce causal features with explicit availability and fit scope.

Deliverables:

- Implement `FeatureSpec`, stateless causal expressions, feature lineage, and warm-up handling.
- Provide lagged returns, rolling volatility, volume statistics, time/session features, and cross-sectional ranks.
- Implement safe point-in-time/as-of joins.
- Implement `FittedTransform` lifecycle and persist fitted state per fold.
- Add prefix-invariance and sentinel-future test helpers reusable by all features.
- Optionally materialize feature sets to versioned Parquet; keep on-demand execution as the first path.

Acceptance criteria:

- Appending or mutating data after `t` cannot change feature output at or before `t`.
- Cross-sectional features use only the complete observation batch available at that decision instant and have a stated missing-member policy.
- Fitted transforms refuse non-training-role data during `fit`.
- Feature manifests identify input dataset, parameters, implementation fingerprint, and fitted state where applicable.

### Stage 3 — Deterministic simulation core

**Goal:** Run an auditable bar-based portfolio simulation with realistic baseline frictions.

Deliverables:

- Implement engine phases, event types, clock, `DecisionContext`, and strategy protocol.
- Implement `OrderIntent`, constraints, `Order`, lifecycle events, and `Fill`.
- Implement subsequent-eligible-bar-open market fills with latency, spread, slippage, fees, lagged-liquidity participation limits, and a missing-bar policy. Do not use the fill bar's eventual volume for an open fill.
- Implement multi-asset cash/position accounting, point-in-time marks, and reconciliation.
- Persist orders, fills, signals, equity, positions, warnings, and cost decomposition.
- Add a simple buy-and-hold/scheduled strategy and one lagged-signal example solely as fixtures and demonstrations.

Acceptance criteria:

- Every fill satisfies `signal_at <= order_at <= eligible_at <= fill_at` and has a full cost decomposition.
- A signal based on a completed bar cannot fill at that bar's open or close.
- Hand-calculated golden scenarios match orders, fills, fees, positions, cash, and equity.
- Permuting asset and input-file order does not change canonical results.
- The ledger reconciles within defined tolerances at every snapshot.
- No-cost and high-cost scenarios differ in the expected direction and exact accounted amount.

### Stage 4 — Evaluation, splits, and experiment registry

**Goal:** Make honest out-of-sample comparison the default workflow.

Deliverables:

- Implement half-open `DataSplit`, rolling and expanding `WalkForwardPlan`, warm-up, purge, and embargo.
- Calculate purge requirements from label horizon; require an explicit override if inference is impossible.
- Fit transforms/models separately inside each training fold.
- Keep trial/validation metrics separate from final test metrics.
- Implement return, volatility, Sharpe-like, drawdown, turnover, exposure, trade, capacity proxy, and cost metrics with declared annualization.
- Implement `RunSpec`, run ID, environment capture, atomic run finalization, failure records, and artifact schema versions.
- Build a rebuildable DuckDB run index and `compare` command.
- Add sensitivity execution across at least base, low-cost, and stressed-cost scenarios.

Acceptance criteria:

- Split tests prove that features and labels cannot cross protected boundaries.
- No fit operation sees validation/test-role rows.
- Identical resolved runs produce identical canonical economic artifacts and hashes.
- A failed run is distinguishable from a complete run and never appears complete in comparisons.
- Comparison output always displays split role, dataset ID, cost model, and warning count alongside performance.

### Stage 5 — CLI workflow and MVP hardening

**Goal:** Deliver a coherent researcher workflow and validate it end to end.

Proposed CLI:

```text
qresearch data ingest --config configs/data.yaml
qresearch data validate --dataset <dataset-id>
qresearch data inspect --dataset <dataset-id>
qresearch features build --config configs/features.yaml
qresearch backtest run --config configs/examples/crypto_momentum.yaml
qresearch walk-forward run --config configs/examples/crypto_momentum.yaml
qresearch runs show <run-id>
qresearch runs compare <run-id> <run-id> ...
qresearch runs index --rebuild
```

Deliverables:

- Thin CLI commands, useful validation/error messages, progress logs, and nonzero error exit codes.
- End-to-end example configurations for crypto and equities using synthetic or redistributable sample data.
- Performance benchmarks representative of a modest local research workload.
- Documentation for timestamp semantics, data contracts, execution assumptions, adding features, and interpreting results.
- Full leakage/realism checklist printed or referenced by each completed run.

Acceptance criteria:

- A new user can execute the documented sample from ingestion through comparison.
- The end-to-end test runs two assets through at least two walk-forward folds.
- A run can be recreated from its artifacts on a clean checkout with access to the referenced dataset.
- Warnings clearly identify survivor-biased universes, assumed spreads, data gaps, stale marks, and other known limitations.
- Benchmarks and memory profiles are recorded before optimization; changes preserve deterministic outputs.

## 5. Recommended implementation order within the simulator

The simulator is the highest-risk component. Implement it through very small scenarios:

1. One asset, two bars, one market order, no costs.
2. Add signal/order/fill latency assertions.
3. Add spread, slippage, and fee decomposition.
4. Add cash and position reconciliation.
5. Add multiple orders and partial/participation-limited fills.
6. Add two assets with simultaneous events and permutation tests.
7. Add missing bars, market sessions, cancels, and end-of-run liquidation policy.
8. Only then connect vectorized features and walk-forward orchestration.

This order makes timing and accounting errors visible before large DataFrames obscure them.

## 6. Configuration model

Use layered, resolved configuration:

```text
DataConfig
UniverseConfig
FeatureConfig
StrategyConfig
ExecutionConfig
PortfolioConfig
SplitConfig
ArtifactConfig
RunConfig (composes the above)
```

Configuration files may be YAML for usability, but are parsed immediately into strict Pydantic models. The fully resolved canonical JSON—not the original YAML—is persisted and hashed. Environment variables are reserved for secrets and machine-specific roots, never hidden economic parameters.

Defaults that improve headline performance are dangerous. Costs should be nonzero by default, same-bar fills should be impossible in the MVP, and missing data should warn or fail rather than silently forward-fill.

## 7. Definition of reproducibility

A reproducible run records:

- canonical `RunSpec` and run ID;
- exact normalized dataset and feature manifest IDs;
- Git commit plus dirty-worktree patch hash when applicable;
- Python/platform and dependency-lock hashes;
- seeds for Python, NumPy, model libraries, samplers, and workers;
- library thread/determinism settings that affect results;
- calendar, symbol map, universe, cost, and execution-model versions;
- artifact schemas, start/end timestamps, status, and warnings.

Byte-identical Parquet is not required because writer metadata may vary. Reproducibility is defined over canonical sorted records and economic values with declared exactness/tolerances. The comparison tool should distinguish exact, tolerance-equal, and different results.

## 8. Research workflow guardrails

The intended workflow is:

1. State a hypothesis, target universe, horizon, primary metric, and cost assumptions.
2. Reserve chronological train, validation, and test periods before exploration.
3. Develop features and fit models using train only.
4. Select parameters using walk-forward validation results.
5. Run the test configuration as a separately labeled promotion step.
6. Persist every material trial, including failures and negative results.
7. Review stability across folds, assets, regimes, and cost stresses—not only aggregate Sharpe.

The software can enforce data access and fit scopes, but it cannot stop a person from repeatedly inspecting test results. Run history and explicit split roles make that behavior visible.

## 9. Performance strategy

Correctness comes first. Initial performance choices:

- Scan only manifest-selected Parquet partitions and columns.
- Use Polars lazy plans and streaming collection where supported.
- Materialize stable, expensive feature sets only after profiling.
- Feed the event loop compact sorted arrays/records rather than Pydantic objects per row.
- Write event outputs in buffered batches.
- Use DuckDB for result aggregation rather than loading every run into Python.
- Avoid multiprocessing until determinism, memory use, and the serial baseline are measured.

Set an MVP benchmark only after representative sample sizes and hardware are agreed. Do not encode an arbitrary throughput promise in the architecture.

## 10. Deferred decisions

Resolve these when implementation reaches the relevant stage:

- Package/lock tool (`uv` is a strong default) and type checker.
- First licensed crypto and equity data providers and exact source availability latency.
- Equity adjustment policy and point-in-time universe source.
- Slippage formula and calibration datasets.
- Short borrow availability/cost; until implemented, short results carry an explicit limitation.
- Whether a single base currency is sufficient beyond MVP.
- Artifact retention and large-signal storage policy.
- Exact annualization and risk-free-rate policies by market/session.

These decisions belong in versioned configuration or short architecture decision records; none should remain an undocumented code default.

## 11. First implementation slice

The recommended first pull request after these documents is intentionally narrow:

- project scaffolding and quality tools;
- UTC timestamp types/validators;
- `Instrument`, `Bar`, and `DatasetManifest` contracts;
- a local Parquet catalog and one synthetic two-asset dataset;
- bar validation and point-in-time query tests;
- documentation of the chosen source bar-label convention.

It should not include strategies or a simulator. Establishing trustworthy data-time semantics first prevents the rest of the system from being built on an ambiguous foundation.
