# Decisions log

Decisions made during implementation that are not fully determined by ARCHITECTURE.md or
DEVELOPMENT_PLAN.md. Each entry says what was decided, why, and what to look at if you
disagree. Newest at the bottom.

Entries marked **autonomous** were made under the "continue through all stages" directive
without a check-in; the rest were reported in conversation at the time.

---

## D1. `uv` + Ruff + mypy `--strict`; `filterwarnings = error`

Plan sec. 10 deferred the tool choice. `uv` was its stated strong default. `mypy --strict`
and warnings-as-errors are the strictest settings that were free to adopt on a greenfield
package; loosening later is cheap, tightening later is not.

## D2. Partition layout gains a `dataset=<id>` level and drops `symbol=`

ARCHITECTURE sec. 4 sketched `.../bar_size=1m/symbol=.../date=...`. Without a dataset
level, a corrected dataset over the same days overwrote the original's files and silently
invalidated its manifest — caught by the "corrections produce a new manifest without
changing old data" acceptance test. The per-symbol level was dropped per the same
section's instruction to compact files. Recorded in ARCHITECTURE sec. 4.

## D3. Derived-bar availability is clamped to the window end

`available_at = max(max(inputs), bar_end)`. An incomplete window (missing last minutes)
would otherwise publish before the window closed, while inputs to it could still arrive.
Recorded in ARCHITECTURE sec. 4 and docs/timestamp_semantics.md.

## D4. `pytz` is a runtime dependency

Initially refused (UTC-only project). Reversed at the user's request: DuckDB needs it for
`fetchall()` and parameter binding on `TIMESTAMPTZ`, and a research tool will be used from
notebooks. `query_asof()` still returns Polars via Arrow as the fast path.

## D5. YAML config loads with `strict=False`; in-process construction stays strict

Enum/Decimal/timedelta/tuple coercion from text is the point of a config file. Shape is
still enforced (`extra="forbid"`). See `application/config.py`.

## D6. One `available_at` per feature row, the max across the feature set

Per-feature availability columns would let a strategy see `ret_1` before `ret_20`. Batch
semantics (ARCHITECTURE sec. 2) argue for one instant per row. A caller wanting a fast
feature earlier computes it in a separate frame.

## D7. Warm-up rows are available when their bar is, with a null value

"Not enough history yet" is a true statement at that instant. Hiding the row would make a
strategy's first decision depend on the longest lookback in the set.

## D8. Rolling windows are positional and nulled across grid gaps by default

A "5-bar return" spanning an overnight gap is a different quantity from the one the name
claims. `require_contiguous=False` opts out per feature.

## D9. Cross-sectional ranks are a second stage with batch availability

Row availability is lifted to the max across the cross-section. Missing-member policy:
null inputs and absent bars are not members; null below `min_members` (default 2). The
denominator varies with gaps until point-in-time universes land.

## D10. Leakage checkers compare floats with `rtol = 1e-9`

Polars' rolling statistics are streaming algorithms; a causal `rolling_std` is not
bit-identical between a prefix and the full series. `rtol=0` is available. Boundary pinned
by tests in both directions.

## D11. Time features are UTC; session-relative features wait for calendars

`MinuteOfDay`/`DayOfWeek` read `bar_start` in UTC. Correct for 24/7 markets, causal but
less meaningful for equities. Session features come with the calendar module.
