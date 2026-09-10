# qresearch

Research and backtesting platform for systematic intraday strategies. Optimised for
trustworthy, reproducible out-of-sample experiments on 1-minute and 5-minute bars —
crypto and liquid US equities first.

Design: [ARCHITECTURE.md](ARCHITECTURE.md). Roadmap: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md).
Timestamp rules: [docs/timestamp_semantics.md](docs/timestamp_semantics.md).

## Status

Stage 1 (data foundation) of the development plan. What exists:

- strict UTC timestamp policy and canonical hashing
- `Instrument`, `Bar`, `DatasetManifest` contracts with timing and OHLC invariants
- local CSV/Parquet adapter with an explicit, required source timestamp-label convention
- immutable, content-addressed Parquet datasets with integrity verification
- point-in-time reads where the availability cutoff is a required argument
- 1m → 5m resampling that cannot publish a coarse bar before its window closes
- cross-row validation (duplicates, gaps, OHLC, availability ordering)
- rebuildable DuckDB views with a point-in-time table macro
- a synthetic two-asset fixture containing gaps, late bars, and a revised duplicate

No strategies, features, or simulator yet — those are Stages 2–3.

## Quickstart

```sh
uv sync --extra dev
uv run qresearch data demo --root data          # generate + ingest the synthetic sample
uv run qresearch data list --root data
uv run qresearch data inspect <dataset-id> --root data
uv run qresearch data head <dataset-id> --root data --as-of 2024-03-04T00:33:00Z
uv run qresearch data sql  <dataset-id> --root data --as-of 2024-03-04T00:33:00Z \
    --where "instrument_id = 'CRYPTO:BTCUSD'"
uv run qresearch data verify <dataset-id> --root data
```

`--as-of` is required on every read command: there is no way to ask for data without
stating when you are allowed to know it. Try `00:33:00Z` on the demo dataset — the BTC
bar for `00:30` is published seven minutes late and will be absent while `00:31` is
present.

Declarative ingestion: `qresearch data ingest -c configs/examples/ingest_synthetic_crypto.yaml`.

## Development

```sh
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

`pytest` runs with `filterwarnings = error`; a deprecation is a failure.

## Layout

```
src/qresearch/
  time.py, ids.py, config.py      UTC policy, canonical hashing, strict model bases
  data/
    contracts.py                  Instrument, Bar, SymbolAlias
    manifests.py                  policies, validation report, DatasetIdentity/Manifest
    catalog.py                    Parquet layout, write, point-in-time scan, verify
    point_in_time.py              BarQuery (as_of required), UnfilteredScan (reason required)
    validation.py                 cross-row checks
    duck.py                       DuckDB views + as-of macro
    synthetic.py                  deterministic fixture generator
    adapters/                     base helpers + local CSV/Parquet adapter
  application/
    ingest.py                     adapter -> validate -> store; resample_bars
    config.py                     YAML -> IngestConfig
  cli.py
```
