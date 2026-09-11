# Data contracts

## Canonical bar

| column | type | notes |
|---|---|---|
| `instrument_id` | string | stable internal id, never a provider ticker |
| `bar_size` | string | `1m`, `5m`, … — one value per dataset |
| `bar_start` | timestamp[us, UTC] | inclusive |
| `bar_end` | timestamp[us, UTC] | exclusive; `bar_end - bar_start == bar_size` |
| `available_at` | timestamp[us, UTC] | earliest consumable instant; `>= bar_end` |
| `open, high, low, close` | float64 | positive; `low <= min(o,c)`, `high >= max(o,c)` |
| `volume` | float64 | unit in the manifest policy |
| `vwap`, `trade_count` | nullable | provider-supported |
| `source`, `source_key`, `revision` | | provenance |
| `ingested_at` | timestamp[us, UTC] | lineage only; no effect on availability |

Natural key: `(instrument_id, bar_size, bar_start, revision)`.

## Dataset identity

`dataset_id = "ds_" + hash(schema_version, asset_class, venue, bar_size, calendar_id,
instrument_ids, range, NormalizationPolicy, normalization_version, source content hashes)`.
Re-ingesting identical bytes under an identical policy yields the same id. Changing any
policy value yields a different dataset, because it produces different data.

`content_digest` hashes canonical row content independent of Parquet writer details.
The same `dataset_id` with a different `content_digest` means normalization is not
deterministic and the catalog refuses to write.

## Normalization policy

| field | meaning |
|---|---|
| `timestamp_label` | `bar_start` or `bar_end`: which edge the source's timestamp denotes. **Required, no default.** |
| `publication_latency` | added to `bar_end` when the source has no `available_at`; zero is flagged as optimistic |
| `price_adjustment` | `none`, `split_only`, `total_return`, `not_applicable` |
| `missing_bars` | `omit` (default) or `error`; never forward-fill |
| `duplicates` | `error` (default), `keep_highest_revision`, `keep_first` |
| `revisions` | `as_known` (default: every revision with its own availability) or `latest_known` |
| `volume_unit` | documented unit |

## Layout

```
<root>/normalized/schema_version=1/asset_class=<c>/bar_size=<s>/dataset=<id>/date=<YYYY-MM-DD>/part-0000.parquet
<root>/manifests/<dataset_id>.json
```

The `dataset=` level makes versions immutable on disk. One file per UTC day holds every
instrument, sorted by `(instrument_id, bar_start, revision)`.

## Reads

`DatasetCatalog.scan_bars(BarQuery)` — `as_of` is required; no row with
`available_at > as_of` is returned, and partitions are pruned on their recorded
`min_available_at`. `scan_all_unfiltered(UnfilteredScan)` exists for manifest building,
integrity checks, and feeding the simulator (which enforces availability per instant);
it requires a stated reason.

DuckDB: `bars_<id>` (plain) and `bars_asof_<id>(as_of)` (filtered) views, rebuilt from
manifests with `qresearch data index`.

## Validation

| check | severity | meaning |
|---|---|---|
| `duplicate_natural_key` | error | |
| `available_before_bar_end` | error | direct look-ahead |
| `bar_size_mismatch` | error | label and interval disagree |
| `ohlc_relationship`, `non_positive_price`, `negative_volume` | error | |
| `zero_volume_bar` | warning | untradeable, not infinitely liquid |
| `availability_out_of_order` | warning | a later bar published before an earlier one |
| `publication_latency_mismatch` | warning | data disagrees with the declared latency |
| `zero_publication_latency` | warning | optimistic timing assumption |
| `missing_bars` | warning/error by policy | gaps in the regular grid (crypto only until sessions are wired into validation) |

Ingestion refuses to write a dataset with any error finding.

## Derived bars

`5m` from `1m`: epoch-aligned half-open windows; `available_at = max(max(inputs),
bar_end)`. See docs/timestamp_semantics.md for why the second term matters.

## Calendars

`24x7:1` (UTC days) and `XNYS:1` (09:30–16:00 America/New_York, computed holidays,
observance rules, 13:00 early closes, ad-hoc closures; verified against published
calendars for 2020–2026). Calendars are versioned: a rule change is a new id.
`attach_sessions(bars, calendar)` adds `session_open`/`session_close`; bars outside any
session get nulls.
