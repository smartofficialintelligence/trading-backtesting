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

## D12. Calendars are versioned rule sets; `XNYS:1` verified for 2020-2026 — **autonomous**

`data/calendars.py` encodes NYSE holidays (with observance rules), 13:00 early closes, and
a list of ad-hoc closures (9/11, Sandy, presidential funerals). Rules are computable for
any year but were only checked against published calendars for 2020-2026; `Juneteenth`
is gated to `>= 2022`. Adding a closure or changing a rule is a new `calendar_id`, never
an edit, because manifests record the id they were validated under. The plan's deferred
"exact source availability latency" and "equity adjustment policy" remain deferred; they
are dataset-policy fields, not calendar fields.

## D13. Session lookup is a data step; session features are expressions — **autonomous**

`attach_sessions(bars, calendar)` adds `session_open`/`session_close` columns; the
session features (`MinutesSinceOpen`, `MinutesToClose`, `SessionFraction`,
`IsEarlyClose`) read them. Forgetting the step is a clear pipeline error rather than a
wrong feature. Bars outside any session carry nulls; that count is the natural input for
the "session errors" validation check (wired in Stage 5 hardening if not sooner).

## D14. Default fill rule: first open at or after eligibility, in a later step — **autonomous**

With contiguous bars, bar N publishes no earlier than `bar_end(N) == bar_start(N+1)`, so
bar N+1's open has already printed when a decision on bar N is made. The engine's phase
order (fills before decisions within an instant) therefore makes **open(N+2)** the
earliest fill for a signal on bar N, under any non-negative latency. This is one bar more
conservative than the conventional "next open" assumption. That convention is available
as `FillRule.OPEN_OF_CURRENT_BAR` — fill at eligibility time at the open of the bar
containing it — because 5-minute research needs it, and every run using it carries an
`optimistic_fill_rule` warning. Golden tests pin both.

## D15. A fill before an instrument's first publication is marked at its own print — **autonomous**

Only possible in the first bar. The open print the execution model traded on is known at
that instant; using it as a provisional mark keeps accounting valid until a close
arrives. It is not exposed to the strategy as a "close".

## D16. Liquidation runs at the first instant at or after the range end — **autonomous**

`EndOfRunPolicy.LIQUIDATE` cancels pending orders and submits flattening orders there,
then keeps processing opens until they fill or expire. If that instant is also an open
event, in-range orders that can fill do so *before* liquidation (phase order). If no
instant exists past the end, positions stay open and `liquidation_impossible` is
recorded. Default is `MARK`: leave positions open at the last point-in-time close.

## D17. Cross-instrument constraints scale by one common factor — **autonomous**

Gross-exposure and cash limits scale every exposure-increasing order proportionally
rather than rejecting in sequence, so the outcome is independent of intent order.
Intents are also sorted canonically before sizing, so permuted intents give identical
ledgers including order ids (tested).

## D18. No liquidity estimate: no participation cap, with a warning — **autonomous**

Before `liquidity_lookback_bars` have published, an order has no trailing-volume
estimate. Rejecting would make the first bars of every run untradeable; capping against
nothing is impossible. The order fills uncapped and `no_liquidity_estimate` is recorded.
Slippage models needing an estimate fall back to `fallback_bps` with `slippage_fallback`.

## D19. Expiry at the same instant as an open expires first — **autonomous**

Conservative tie-break. Likewise a stale-mark check runs at decisions, not during gaps
where nothing is valued; a single instrument's gap therefore produces no warning, a
multi-instrument gap does (tested).

## D20. Shorts carry no borrow cost or availability constraint — **autonomous**

Plan sec. 10 deferred borrow modelling. `allow_short=False` is the default; when
enabled, a `short_position` warning is recorded on every run that opens one.
