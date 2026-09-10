# Architecture

## 1. Purpose and scope

This project is a research and backtesting platform for systematic intraday strategies. Its first users are researchers working with 1-minute and 5-minute bars for crypto and liquid US equities. It is optimized for trustworthy, reproducible out-of-sample experiments rather than live execution.

The first version deliberately favors a small number of explicit, testable components. It is not a generic event-sourcing framework, a live-trading system, or a distributed compute platform.

### Design priorities

In priority order:

1. Prevent future information from reaching a strategy or fitted transform.
2. Make execution assumptions visible and conservative.
3. Reproduce a result from immutable inputs and a frozen configuration.
4. Support portfolios of assets without results depending on iteration order.
5. Make common feature and ML workflows convenient.
6. Preserve clear extension points for richer data and execution models.

## 2. Architectural principles

### Point-in-time correctness is a data contract

Every observation that can influence a decision has an `available_at` timestamp: the earliest instant at which the observation is allowed to be consumed in the simulation. Event time alone is not sufficient.

For a completed bar:

- `bar_start` and `bar_end` describe the market interval.
- `available_at` describes when the completed bar becomes visible to research code.
- `ingested_at` records when this copy arrived in our storage and supports data lineage; it does not make the value historically available.

A bar is normally available no earlier than `bar_end`, plus a configured publication/data latency. A feature derived from several observations is available no earlier than the maximum `available_at` of all its inputs plus any declared computation latency.

The engine enforces:

```text
input.available_at <= decision_at
signal_at <= order_at <= eligible_at <= fill_at
```

`signal_at` is the strategy decision timestamp. `order_at` is when the validated order is submitted to the simulated execution gateway. `eligible_at` is when it reaches the simulated venue after order latency. `fill_at` is the modeled transaction time. These fields must never be collapsed into one generic timestamp.

All timestamps are timezone-aware UTC at system boundaries and stored as UTC. Naive datetimes are rejected. Parquet timestamps use a documented microsecond UTC representation unless a source requires finer precision.

### Strategy code cannot access the raw future

Strategies receive a read-only `DecisionContext` containing only the current point-in-time feature slice, portfolio state, and clock. They do not receive a full DataFrame, a catalog handle, row offsets into future data, or the execution model.

The execution model may consume a later market event once the simulation clock reaches that event in order to price an order that was already eligible. That access is isolated from strategy and feature code. For example, an order created after minute `t` closes may fill at the open of the first subsequent eligible minute; only the opening event—not that minute's eventual high, low, close, or volume—is usable for an open fill.

### Batch simultaneous information

At a given timestamp, all newly available observations are published as one batch before a strategy is called. The strategy produces targets/orders for the portfolio, not one asset at a time. Stable sorting is used for deterministic bookkeeping, but changing asset sort order must not change economic results.

### Immutable inputs and append-only results

Raw and normalized datasets are versioned by manifests and checksums. A run references exact dataset versions, configuration, code revision, dependency lock hash, and random seeds. Completed run artifacts are append-only; rerunning an identical specification may reuse the same content-addressed identifier but must not silently overwrite a different result.

### Vectorize calculations; simulate state transitions

Polars lazy expressions are used for normalization and feature computation. DuckDB is used for ad hoc analytical queries and result comparison. Portfolio accounting, order lifecycle, and fills remain a chronological state machine because vectorization can hide timing errors.

## 3. System overview

```text
Provider adapters
      |
      v
Raw immutable Parquet ---> validation reports
      |
      v
Canonical normalized Parquet + dataset manifests
      |
      +----> DuckDB catalog/views (discovery and analytical queries)
      |
      v
Point-in-time loader ---> feature pipeline ---> chronological simulator
                                                  |       |       |
                                             strategy  execution portfolio
                                                  \       |       /
                                                   run events
                                                       |
                                                       v
                                              metrics and artifacts
                                                       |
                                                       v
                                              experiment comparison
```

The major packages are:

- **Data**: provider adapters, schemas, normalization, quality checks, dataset manifests, and point-in-time loading.
- **Features**: causal feature definitions, fitted transforms, materialization, and lineage.
- **Simulation**: deterministic clock, market-data publication, order lifecycle, execution, costs, accounting, and constraints.
- **Research**: dataset splits, walk-forward orchestration, parameter trials, metrics, and experiment comparison.
- **Artifacts**: frozen run specifications, manifests, event ledgers, reports, and run index.
- **CLI**: thin commands that compose application services. Business logic does not live in command handlers.

## 4. Data architecture

### Storage zones

Data is stored under a configurable root, outside the Python package:

```text
data/
  raw/provider=.../asset_class=.../date=YYYY-MM-DD/*.parquet
  normalized/schema_version=1/asset_class=.../bar_size=1m/dataset=<dataset_id>/date=YYYY-MM-DD/*.parquet
  features/feature_set=.../version=.../date=.../*.parquet
  manifests/<dataset_id>.json
runs/<run_id>/...
```

Raw data is retained exactly as received where licensing permits. Normalized data uses canonical identifiers and schemas. Partition columns are chosen for pruning; files should be compacted to useful analytical sizes rather than producing one tiny file per symbol-minute. The `dataset=<dataset_id>` level is what makes dataset versions immutable on disk: without it, a corrected version of the same instruments over the same days would overwrite the original's files and silently invalidate every manifest and run that referenced them. There is no per-symbol level; one file per UTC day holds every instrument sorted by `(instrument_id, bar_start)`, so row-group statistics still prune by instrument without multiplying file count by universe size.

DuckDB is a query/catalog layer over Parquet, not the source of truth. Its views and local database can be rebuilt from manifests.

### Canonical bar schema

The initial bar contract contains:

| Field | Meaning |
|---|---|
| `instrument_id` | Stable internal identifier, separate from a provider ticker |
| `bar_size` | Canonical duration such as `1m` or `5m` |
| `bar_start` | Inclusive UTC interval start |
| `bar_end` | Exclusive UTC interval end |
| `available_at` | Earliest UTC time the completed record can be consumed |
| `open`, `high`, `low`, `close` | Canonically adjusted or unadjusted prices per dataset policy |
| `volume` | Base/share volume with documented unit |
| `vwap`, `trade_count` | Nullable provider-supported fields |
| `source` | Provider and feed identifier |
| `source_key` | Provider record identity where available |
| `revision` | Source revision/version |
| `ingested_at` | UTC ingestion timestamp |

Invariants include unique `(dataset_version, instrument_id, bar_size, bar_start)`, `bar_start < bar_end <= available_at`, nonnegative volume, valid OHLC relationships, and a complete declaration of session/calendar behavior.

Five-minute bars derived from 1-minute data use half-open, epoch-aligned windows. Their `available_at` is the maximum input availability and never earlier than the window's own `bar_end`. The second bound handles incomplete windows: if the final minutes are absent, the slowest present input may be available well before the window closes, and publishing then would offer a coarse bar that later inputs to the same window could still change.

### Instruments, symbols, calendars, and universes

`Instrument` provides a stable identity and time-varying symbol mappings. Equity sessions use versioned exchange calendars; crypto declares a 24/7 calendar and venue. Prices and quantities retain declared currencies and units.

A `UniverseDefinition` is evaluated point in time. A current list of liquid assets must not be applied retroactively. Early MVP experiments may use a static, explicitly labeled survivor-biased universe, but reports must carry that limitation.

Corporate actions are effective-dated observations with their own `available_at`. Adjustment policy is recorded in the dataset manifest. Point-in-time ranking and signals must not use adjustment factors learned from future actions.

### Dataset manifests

Every normalized dataset version has a `DatasetManifest` containing:

- immutable `dataset_id` and schema version;
- provider, asset class, venue, bar size, date range, instruments, and calendar version;
- partition file paths, row counts, hashes, and min/max timestamps;
- normalization code version and source-data identifiers;
- adjustment, missing-bar, duplicate, revision, and timestamp-label policies;
- validation results and creation timestamp.

Provider corrections create a new dataset version. Existing experiments remain attached to the older manifest.

## 5. Time and event semantics

### Engine phases

For each UTC clock instant, the simulator applies deterministic phases:

1. Activate orders whose `eligible_at` has arrived.
2. Process executable market openings/ticks and create fills for previously eligible orders.
3. Apply fills, fees, cash movements, and position changes.
4. Publish observations whose `available_at` has arrived.
5. Update causal features from the newly visible batch.
6. Mark the portfolio using the configured point-in-time valuation policy.
7. Invoke the strategy once with the complete simultaneous batch.
8. Validate emitted intents, apply constraints, and schedule orders with explicit latency.
9. Append events and snapshots to the run ledger.

No order emitted in phase 7 can fill in an earlier phase at the same timestamp unless an execution model explicitly supports a later intra-timestamp sequence backed by suitable quote/trade data. The bar-based MVP does not.

When data have identical timestamps, ordering is fixed by event phase and stable identifiers, never by arbitrary DataFrame or filesystem ordering.

### Bar-based execution limitations

OHLCV bars do not reveal the path within a bar, queue position, displayed liquidity, or whether a limit and stop were touched in which order. The MVP therefore supports conservative market-order fills at the first eligible subsequent bar open. It does not claim realistic limit-order simulation.

The modeled fill price is conceptually:

```text
reference price
+ side * half_spread
+ side * slippage(reference price, quantity, point-in-time liquidity, volatility)
```

Fees are separate cash ledger entries. Spread, slippage, commissions, latency, and participation caps are explicit versioned configurations. For an open fill, liquidity and volatility estimates must have been available by `order_at`; the just-opened bar's eventual volume cannot be used. If quotes are unavailable, spread is an assumption and is labeled as such in results.

Missing next bars delay or reject fills according to an explicit policy. They never cause a fill using a stale price without a recorded warning.

## 6. Main domain objects

Pydantic models are used for configuration, persisted contracts, and validation at boundaries. Frozen standard dataclasses or small typed records may be used inside performance-critical loops after validation. Monetary/accounting calculations must use a documented numerical policy; floats are acceptable for market analytics, while tolerances and reconciliation rules are explicit.

### Market and reference data

#### `Instrument`

Stable `instrument_id`, asset class, venue, quote/base currency, price and quantity increments, timezone/calendar identifier, and effective-dated symbol aliases.

#### `Bar`

Canonical fields described above. Its identity and timing fields are immutable.

#### `Observation[T]`

Generic envelope for later quotes, trades, funding, news, corporate actions, and order-book snapshots:

```python
Observation(
    instrument_id: InstrumentId,
    event_at: datetime,
    available_at: datetime,
    source: str,
    payload: T,
)
```

`event_at` says when the underlying event occurred; `available_at` says when research code may use it.

#### `DatasetManifest`

Immutable lineage and quality contract for an exact set of Parquet files.

#### `UniverseDefinition` / `UniverseSnapshot`

A rule plus point-in-time inputs, and the resulting membership/effective interval. A snapshot carries the reason each instrument entered or left.

### Features and strategies

#### `FeatureSpec`

Name, version, input columns/features, parameters, lookback requirement, null/warm-up policy, declared output type, and implementation fingerprint. Feature outputs include `as_of`/`available_at` semantics and lineage.

#### `FittedTransform`

An ML-style stateful transform with `fit(train_data)`, `transform(as_of_data)`, and serializable fitted state. The orchestrator, not the transform, controls which split is supplied to `fit`.

#### `DecisionContext`

A read-only snapshot containing `decision_at`, the eligible universe, current point-in-time features, recent data only when explicitly requested by lookback, positions, cash, outstanding orders, and fold/run identifiers.

#### `Signal`

A diagnostic research output: strategy ID, instrument, `signal_at`, numeric value or class, horizon, and provenance. Signals are not fills and do not directly mutate the portfolio.

#### `OrderIntent`

The strategy's requested action, preferably a target position/weight for simple portfolio strategies. It records `created_at` (equal to the decision time), instrument, side/target, sizing information, time in force, and strategy metadata.

#### `Strategy`

A deliberately small interface:

```python
class Strategy(Protocol):
    def on_decision(self, context: DecisionContext) -> Sequence[OrderIntent]: ...
```

Model fitting and hyperparameter selection occur outside `on_decision`. Strategy state must be initialized/reset per fold and be serializable or reconstructible from the run specification.

### Orders and execution

#### `Order`

Validated instruction produced from an intent: immutable order ID, instrument, side, quantity, order type, `signal_at`, `order_at`, `eligible_at`, time in force, status, and parent strategy/run IDs. State transitions are appended as `OrderEvent`s rather than rewriting history.

#### `Fill`

Order ID, instrument, side, quantity, `fill_at`, reference price, spread impact, slippage impact, final price, fee components, execution-model version, and liquidity metadata. The decomposition makes cost assumptions auditable.

#### `ExecutionModel`

```python
class ExecutionModel(Protocol):
    def match(
        self,
        market_event: MarketEvent,
        eligible_orders: Sequence[Order],
        state: ExecutionState,
    ) -> Sequence[Fill]: ...
```

It alone may inspect fields used to model a fill. It must not call strategy code or mutate portfolio accounting.

#### `CostModel`

Computes commissions, exchange/regulatory fees, spread and slippage components under a versioned configuration. Separating the interface allows sensitivity tests, though the MVP can implement it as a component of the bar execution model.

### Portfolio and controls

#### `Position`, `CashBalance`, `PortfolioSnapshot`

Point-in-time position quantities, cost basis, realized/unrealized P&L, cash by currency, equity, gross/net exposure, and valuation timestamp. Snapshots are derived from an immutable fill/cash ledger and periodically reconciled.

#### `Portfolio`

Applies fills and cash events and produces snapshots. Accounting is independent of strategy and metrics.

#### `ConstraintSet`

Transforms or rejects intents using maximum gross/net exposure, per-instrument position, leverage, shorting, participation, and cash rules. Every rejection or resize is recorded.

### Research and experiments

#### `TimeRange` and `DataSplit`

Half-open UTC intervals `[start, end)` with a role of train, validation, test, warm-up, purge, or embargo. Splits must not overlap after considering label horizons and feature lookbacks.

#### `WalkForwardPlan`

Ordered folds, each with train/validation/test ranges, purge/embargo rules, refit schedule, and universe policy. Test results are kept separate from parameter selection.

#### `RunSpec`

Frozen resolved configuration: dataset IDs, feature/strategy/execution versions and parameters, fold, seed set, code revision, dependency hash, and output schema version. Secrets and machine-specific paths are excluded from its canonical hash.

#### `RunResult`

Run identity and status plus artifact references, warnings, metrics, and reproducibility metadata. Detailed events remain in Parquet rather than being embedded in the model.

#### `Experiment`

A named group of comparable runs with one research question, shared search space, primary metric, and selection rule. It distinguishes development/validation comparisons from final test evaluation.

## 7. Principal interfaces and ownership

The initial interfaces should stay narrow:

```python
class MarketDataSource(Protocol):
    def ingest(self, request: IngestRequest) -> RawDatasetRef: ...


class DatasetCatalog(Protocol):
    def resolve(self, dataset_id: str) -> DatasetManifest: ...
    def scan_bars(self, query: BarQuery) -> pl.LazyFrame: ...


class FeaturePipeline(Protocol):
    def fit(self, training_query: PointInTimeQuery) -> FittedFeaturePipeline: ...
    def transform(self, query: PointInTimeQuery) -> pl.LazyFrame: ...


class BacktestEngine(Protocol):
    def run(self, spec: RunSpec) -> RunResult: ...


class Splitter(Protocol):
    def split(self, range: TimeRange) -> Sequence[WalkForwardFold]: ...


class ArtifactStore(Protocol):
    def begin(self, spec: RunSpec) -> RunHandle: ...
    def finalize(self, handle: RunHandle, result: RunResult) -> None: ...
```

Concrete local-filesystem implementations are sufficient for the MVP. Dependency injection is done through constructors/application services, not a framework container.

## 8. Backtest outputs and experiment registry

Each run directory contains at least:

```text
runs/<run_id>/
  run_spec.json
  environment.json
  status.json
  metrics.json
  warnings.json
  orders.parquet
  fills.parquet
  positions.parquet
  equity_curve.parquet
  signals.parquet          # optional, configurable due to size
  fold_predictions.parquet # when applicable
```

A small DuckDB experiment index exposes runs and metrics for comparison. It is derived from immutable run artifacts and can be rebuilt. Schema versions permit artifact migration without mutating historical runs.

Required baseline reports include return and risk metrics, turnover, exposures, trade/fill counts, hit rate, drawdown, cost decomposition, data-coverage warnings, results by asset/day/session, and train/validation/test labels. Metrics must state annualization and session assumptions; crypto and equities cannot share an unexplained annualization constant.

## 9. Leakage and realism risk register

These are the highest-risk failure modes and their controls.

| Risk | Typical failure | Required control |
|---|---|---|
| Bar timestamp ambiguity | Using a bar labeled `10:00` at its start even though it represents data through `10:01` | Canonical start/end plus validated `available_at`; provider-specific mapping tests |
| Same-bar execution | Compute from a close and fill at that close or earlier open | Phase-ordered engine; default first eligible subsequent-bar-open fills |
| Cross-sectional asynchrony | At 10:01, use one asset's delayed 10:00 bar as though it arrived with all others | Per-observation `available_at`; point-in-time batch joins |
| Unsafe joins | Backward/forward joins accidentally select a future fundamental, quote, or feature | Central as-of join utility requiring right-side `available_at <= decision_at`; adversarial tests |
| Rolling window errors | Centered windows, negative shifts, or global calculations bleed future rows | Approved causal Polars expressions and prefix-invariance tests |
| Resampling leakage | A partial 5-minute bar is treated as complete | Half-open windows; output availability from the latest input |
| Global preprocessing | Scalers, imputers, PCA, encoders, or feature selection fit across validation/test | Fold-owned fitted artifacts; fit API accepts only train-role data |
| Label overlap | Training labels extend into validation/test | Purging by maximum label horizon; embargo where repeated/overlapping samples require it |
| Hyperparameter test peeking | Repeatedly select strategies on test performance | Experiment roles and explicit one-way promotion; test is not an optimization input |
| Survivorship bias | Use today's equity/crypto universe historically | Point-in-time membership; prominently label static-universe experiments |
| Corporate actions | Apply future split/dividend knowledge or mix adjusted signals with raw execution prices | Versioned effective-dated actions and one declared adjustment policy |
| Delistings and missing data | Drop failed assets or forward-fill through halts | Preserve inactive instruments; explicit missing/halt/delist policies |
| Revised data | Backtest against a corrected history that was unavailable then | Immutable source revisions and manifest IDs; distinguish latest-known from as-known datasets |
| Intrabar path assumptions | Assume both a stop and target fill in favorable order from OHLC | Do not support in MVP; later use conservative ambiguity policy or finer data |
| Infinite liquidity | Fill large orders at mid/open regardless of capacity | Size-dependent slippage and caps based on lagged point-in-time liquidity; rejections/partial fills |
| Free trading | Ignore fees, spread, borrow, funding, and market impact | Nonzero explicit costs by default; cost decomposition and sensitivity runs |
| Unrealistic latency | Signal, order, and fill share one timestamp | Separate timestamps and configurable data/order latency |
| Stale valuation/FX | Mark all assets with future or stale prices without disclosure | Point-in-time marks, maximum staleness, explicit FX source |
| Session errors | Trade equities outside sessions or across DST incorrectly | Versioned exchange calendars and UTC conversions tested around DST/holidays |
| Random/non-deterministic runs | Threading, seeds, unordered files, or unstable tie breaks change output | Stable ordering, centralized seeds, environment manifest, reproducibility test |
| Multiple comparisons | Report the best of many trials without accounting for search | Persist all trials, predeclare primary selection metric, report trial count and robustness |

No architecture can guarantee “absolutely no look-ahead bias” by assertion. The practical standard is defense in depth: restrictive interfaces, invariants, adversarial tests, provenance, and reviewable artifacts.

## 10. Testing strategy

Tests are organized around invariants rather than only examples:

- **Schema tests** reject naive timestamps, invalid OHLC, duplicate keys, and `available_at < bar_end`.
- **Provider contract tests** map known source timestamps and sessions to canonical bars.
- **Prefix-invariance tests** assert that appending future rows cannot change earlier feature values or decisions.
- **Sentinel-future tests** inject extreme future values and assert earlier outputs are unchanged.
- **Engine timeline tests** prove order eligibility and fills respect phase/timestamp inequalities.
- **Multi-asset permutation tests** randomize asset/input order and require identical results.
- **Accounting tests** reconcile cash, positions, fees, equity, and P&L after every event sequence.
- **Split tests** prove no overlap after lookback, label horizon, purge, and embargo.
- **Cost tests** verify direction and exact decomposition for buys/sells and tier boundaries.
- **Golden scenario tests** use tiny hand-calculated datasets with known orders, fills, and equity.
- **Reproducibility tests** run an identical specification twice and compare canonical artifacts/hashes.

Property-based tests are valuable for timestamps, event order, splits, and ledgers, but can be introduced after the deterministic examples are in place.

## 11. Extension path

The generic `Observation` envelope and narrow interfaces permit later additions without redesigning the core:

- quote/trade and order-book events can replace bar-based spread and execution assumptions;
- funding events can enter the cash ledger with effective and available timestamps;
- options add instrument definitions, chains, corporate-action handling, and surface snapshots;
- news adds event, publication, revision, and receipt timestamps;
- ML models implement fitted strategy/feature artifacts with fold-owned training;
- a future live system may reuse schemas and strategies, but should use a separate runtime and risk architecture rather than turning the research engine into an order router.

## 12. Explicit non-goals for the initial system

- Live trading, broker connectivity, or production order routing.
- Tick-accurate or queue-position execution simulation.
- Realistic limit, stop, or complex order behavior from OHLCV alone.
- Distributed execution or a service-oriented architecture.
- A web UI, multi-user authorization, or hosted experiment service.
- A general plugin framework.
- Options, order books, news, and alternative data in the MVP.

## 13. Initial architectural decisions

1. Use a chronological event/state simulator with vectorized Polars feature preparation.
2. Use local Parquet as truth and DuckDB as a rebuildable catalog/query/index layer.
3. Make `available_at` mandatory for every decision input.
4. Default to subsequent eligible bar-open market fills with explicit costs and latency.
5. Use Pydantic at boundaries and simple typed Python objects internally.
6. Store full event ledgers and resolved run specifications for auditability.
7. Implement one strong local path before introducing registries, plugins, or distributed systems.

