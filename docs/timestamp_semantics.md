# Timestamp semantics

This document is the reference for what every timestamp in the system means. It is
normative: code that disagrees with it is wrong.

## The policy in one paragraph

All timestamps are timezone-aware UTC. Naive datetimes are rejected at every boundary;
aware datetimes in other zones are converted. Every observation that can influence a
decision carries an `available_at` — the earliest instant research code may consume it —
and that field, not the event time, is what point-in-time reads filter on. Reads that
skip the filter exist, but are separately named and must state a reason.

## Bar timestamps

A completed bar has three timestamps and one lineage field:

| Field | Meaning | Constraint |
|---|---|---|
| `bar_start` | Inclusive start of the market interval | |
| `bar_end` | Exclusive end of the market interval | `bar_start < bar_end`, and `bar_end - bar_start == bar_size` |
| `available_at` | Earliest instant the completed bar may be consumed | `available_at >= bar_end` |
| `ingested_at` | When *our copy* arrived on disk | No bearing on availability |

Intervals are half-open: a `1m` bar labelled `bar_start = 10:00` covers `[10:00, 10:01)`.

`ingested_at` deliberately has no effect on availability. Backfilling five years of data
today must not make all of it "available" as of today; historical availability is a
property of the source, recorded in `available_at`.

## The source label convention

Providers label bars by one edge or the other, and never say which in the data itself.
This is the single most common source of a one-bar look-ahead, so every adapter must
declare it explicitly via `NormalizationPolicy.timestamp_label`:

| `timestamp_label` | Source `10:00` means | Resulting interval |
|---|---|---|
| `bar_start` | "the minute beginning at 10:00" | `[10:00, 10:01)` |
| `bar_end` | "the minute ending at 10:00" | `[09:59, 10:00)` |

There is no default. The two conventions yield different `dataset_id`s from the same
bytes (see `tests/contracts/test_timestamp_label.py`), so a mislabelled ingestion can
never alias or overwrite a correct one.

**How to determine a provider's convention.** Do not guess from documentation alone.
Take a bar with a large, well-known move (an exchange open, a scheduled announcement) and
check which label places it in the interval where it actually happened. Record the
evidence in the adapter's contract test.

### Known conventions

| Source | Label | Notes |
|---|---|---|
| `synthetic:bars-v1` | `bar_start` | Test fixture; also emits `available_at` and `revision` columns. |

Add a row for every real provider adapter as it lands.

## Availability

`available_at` is either provider-published (preferred, when a source has such a column)
or derived as `bar_end + publication_latency`. The latency is part of the dataset
identity. Zero latency is representable but flagged by validation as optimistic: no real
feed publishes a bar the instant its interval closes.

Availability can run backwards. A late-published bar N may become available after bar
N+1. That is real, and the loader represents it honestly: at a cutoff between the two
publications, a query returns bar N+1 and *not* bar N — a hole in the middle of the
series, not a truncation. Every rolling feature must tolerate this.

## Derived bars

A coarser bar assembled from finer ones (`5m` from `1m`) has:

- `bar_start` on the epoch-aligned grid for its size. `5m` windows always start at
  `:00`, `:05`, …; two datasets over the same period bucket identically.
- `available_at = max(max(inputs.available_at), bar_end)`.

The second term is the one that is easy to miss. It handles an *incomplete* window: if
the last minutes of a `5m` window are absent, the slowest present input might be
available well before the window ends. Publishing then would show a consumer a coarse bar
that is not final — inputs for that same window could still arrive and change it.
Clamping to `bar_end` means a coarse bar is only ever offered once no further input can
belong to it.

## Decision and execution timestamps

The simulator (not yet built) will enforce a second chain:

```
input.available_at <= decision_at
signal_at <= order_at <= eligible_at <= fill_at
```

| Field | Meaning |
|---|---|
| `decision_at` | The instant a strategy is invoked; it sees only inputs with `available_at <= decision_at` |
| `signal_at` | The strategy's decision timestamp; equals `decision_at` |
| `order_at` | Validated order submitted to the simulated gateway |
| `eligible_at` | Order reaches the simulated venue after order latency |
| `fill_at` | Modelled transaction time |

These are never collapsed into one field. See `ARCHITECTURE.md` §2 and §5.

## Storage representation

Parquet `timestamp[us, UTC]` throughout. Polars `Datetime("us", "UTC")`. DuckDB
`TIMESTAMPTZ`. Python `datetime` with `tzinfo=UTC`. Canonical JSON renders instants as
ISO-8601 with a trailing `Z`.
