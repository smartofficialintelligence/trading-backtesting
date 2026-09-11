# qresearch

Research and backtesting platform for systematic intraday strategies. Optimised for
trustworthy, reproducible out-of-sample experiments on 1-minute and 5-minute bars —
crypto and liquid US equities first.

- Design: [ARCHITECTURE.md](ARCHITECTURE.md) · Roadmap and status: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
- Timestamps: [docs/timestamp_semantics.md](docs/timestamp_semantics.md) · Data: [docs/data_contracts.md](docs/data_contracts.md)
- Execution: [docs/execution_assumptions.md](docs/execution_assumptions.md) · Workflow: [docs/research_workflow.md](docs/research_workflow.md)
- Before believing a result: [docs/leakage_checklist.md](docs/leakage_checklist.md)
- Judgment calls made during implementation: [docs/decisions.md](docs/decisions.md)

## Status

The MVP defined in the development plan (Stages 0–5) is implemented. Not live trading.

| layer | what exists |
|---|---|
| providers | **Binance spot klines** (real, live-verified) and a local CSV/Parquet adapter; timestamp label proven against raw trades |
| data | immutable content-addressed Parquet datasets; required bar-label convention; point-in-time reads with a mandatory `as_of`; validation with a calendar-aware session check; 1m→5m resampling; DuckDB views |
| calendars | `24x7:1` and `XNYS:1` (holidays, observance, early closes, DST) |
| features | causal expressions with derived availability; gap-aware windows; cross-sectional ranks with batch availability; session features; fold-owned fitted transforms; leakage checkers in the package |
| simulation | phase-ordered bar-open engine; distinct signal/order/eligible/fill timestamps; decomposed spread/slippage/fee; participation caps; reconciled accounting; order-independent constraints |
| research | walk-forward folds with stated purge and embargo; labelled annualisation; per-fold and stitched per-role metrics; cost-scenario sensitivity |
| jobs | background CLI subprocesses with progress parsed from `--json-logs`, bounded queue, cancellation, crash reaping; `qresearch jobs submit/list/show/cancel` |
| ui | local app (`qresearch ui`, `--extra ui`): filterable run list, overlaid equity curves with assumptions above performance, per-run report, JSON ledger API |
| reporting | per-run self-contained `report.html` (inline SVG, no server/CDN): assumptions and warnings above the metrics, equity/drawdown/exposure, cost attribution, trade P&L distribution, fold ribbon, stability table |
| artifacts | run id = hash of the resolved spec; atomic publish; identical rerun reuses; divergent rerun kept aside; environment capture; rebuildable DuckDB run index; `runs reproduce` |

## Quickstart

```sh
uv sync --extra dev

uv run qresearch data demo --root data                     # synthetic 2-asset crypto sample
uv run qresearch data ingest -c configs/examples/ingest_binance_btc_eth.yaml --root data   # real data
uv run qresearch data inspect <dataset-id> --root data
uv run qresearch data head <dataset-id> --root data --as-of 2024-03-04T00:33:00Z

# put the dataset id into the example config, then:
uv run qresearch backtest run -c configs/examples/crypto_momentum.yaml --root data --runs runs
uv run qresearch runs list --runs runs
uv run qresearch runs compare <run-a> <run-b> --runs runs --role test
uv run qresearch runs report <run-id> --runs runs      # report.html (also written automatically)
uv run qresearch ui --runs runs --root data           # local UI on http://127.0.0.1:8000
uv run qresearch jobs submit -c configs/my_first_backtest.yaml --root data --runs runs --wait
uv run qresearch runs reproduce <run-id> --root data --runs runs
```

`--market equity` on `data demo` produces an XNYS-session sample for
`configs/examples/equities_mean_reversion.yaml`.

Two things to internalise before reading any number: the default fill rule is the
textbook "next open", which is optimistic by a few seconds and always flagged
`optimistic_fill_rule` — every run also produces the conservative open(N+2) twin, and the
pair brackets the truth; and every read requires `--as-of`. Both are explained in the
docs above.

## Development

```sh
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
uv run python scripts/benchmark.py --instruments 5 --days 5     # baseline timings
```

`pytest` runs with `filterwarnings = error`; a deprecation is a failure. Live-venue tests
are deselected by default — `QRESEARCH_NETWORK_TESTS=1 uv run pytest -m network` to run
them.

The suite includes a **differential test against `backtesting.py`** (`-m oracle`, needs
`--extra oracle`): the same strategy on the same bars through an independent engine, which
must produce identical fills. On a day of real BTCUSDT data it agrees exactly — 718 fills,
zero price difference, final equity matching to float64 rounding. It exists to catch
engine drift that our own tests would happily agree with.

## Layout

```
src/qresearch/
  time.py, ids.py, config.py            UTC policy, canonical hashing, strict model bases
  data/                                 contracts, manifests, catalog, validation, calendars,
                                        synthetic fixtures, DuckDB views, adapters/
  features/                             contracts, pipeline, technical, session, cross_sectional,
                                        transforms, leakage checkers, registry
  simulation/                           clock, engine, execution, portfolio, constraints, events
  strategy/                             DecisionContext + Strategy protocol, examples, registry
  research/                             splits, walk_forward, metrics, experiments (index/compare)
  artifacts/                            RunSpec/RunResult, environment capture, local store
  application/                          ingest, run_backtest (orchestration), config loaders
  cli.py
tests/                                  unit/, contracts/, integration/, golden/ (hand-calculated)
configs/examples/                       ingest and backtest YAML examples
docs/                                   semantics, contracts, assumptions, workflow, checklist, decisions
scripts/benchmark.py
```
