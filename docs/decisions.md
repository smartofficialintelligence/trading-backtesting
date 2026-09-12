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

## D14. Default fill rule: first open at or after eligibility, in a later step — **autonomous** — *superseded by D35*

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

## D21. Splits assign rows by `available_at`; embargo is a training exclusion — **autonomous**

A split is a period of knowledge. A bar whose interval is in the training range but which
published after the cutoff belongs to the period it published in. Embargo ranges (after
each fold's test) are recorded and subtracted from later folds' training predicates
rather than modelled as contiguous ranges, since a chronological walk-forward's next
training window legitimately overlaps the previous test.

## D22. The purge must be stated — **autonomous**

`WalkForwardPlan` refuses `purge == 0` unless `label_horizon` is given (even as zero).
Effective purge is `max(purge, label_horizon)` and is applied both after training and
after validation.

## D23. `run_id` hashes everything economic, including `code_revision`; not the label — **autonomous**

A run under a dirty worktree gets `<sha>-dirty-<diff hash>`. Renaming a run does not make
it a new run; changing the code, a feature's fingerprint, the cost scenario, the seed, or
the experiment id does.

## D24. Identical spec: reuse; divergent economics: keep aside and raise — **autonomous**

`LocalArtifactStore.finalize` compares the economic digest (fills + equity curve, sorted,
bookkeeping columns dropped) against an existing run of the same id. Equal → the new
directory is discarded and the existing result returned. Different → the new directory
moves to `conflicts/` and `ReproducibilityError` is raised. The original is never
overwritten. Failed runs go to `failed/` and never appear in `list_runs()`.

## D25. Per-role aggregates stitch fold curves — **autonomous**

Each fold's simulation starts from `initial_cash`. The aggregate curve for a role scales
each fold's equity so it starts where the previous fold ended, then re-scales to the
first fold's starting equity. Metrics on that curve are the "walk-forward test" numbers;
per-fold metrics are kept alongside for stability review.

## D26. `period_hit_rate` is a period-level proxy — **autonomous**

Fraction of bar periods with positive return among periods that began with non-zero gross
exposure. Round-trip trade attribution is not attempted in the MVP; the metric is named
to say what it is.

## D27. Cost scenarios are named presets, part of run identity — **autonomous**

`base`, `free`, `low`, `stressed`. A sensitivity run is one run per scenario under the
same experiment id. Custom presets belong in configuration when the need arises.

## D28. Manifests carry `Instrument` definitions outside the identity hash — **autonomous**

The simulator needs quantity increments and the calendar; the catalog previously stored
only ids. Definitions are stored on the manifest but excluded from `dataset_id` since an
alias correction is not a data change. Manifests written before this field default to
empty and are refused by the orchestrator with a re-ingest message.

## D29. `runs reproduce` re-executes from the stored spec; exit 3 on divergence — **autonomous**

Reproduction is defined over the economic digest (sorted fills and equity curve), not
bytes. A divergence exits non-zero so CI or a promotion script can gate on it.

## D30. `features build` writes a Parquet table plus a JSON sidecar — **autonomous**

The sidecar records the dataset id, each feature's spec and fingerprint, and the row
count. Every row keeps `available_at`; the file is for inspection and ML table export,
not a cache the run reads (on-demand computation stays the primary path per plan §Stage 2).

## D31. Synthetic equity sample is session-only, 2024-03-04/05 (EST) — **autonomous**

`SyntheticSpec.equity()` generates bars only inside XNYS sessions so the equity example
runs through session features and XNYS annualisation end to end. Dates precede the 2024
DST change deliberately; the DST paths are covered by calendar unit tests.

## D32. `outside_session` is a warning, not an error — **autonomous**

Extended-hours prints in an equity dataset are common and sometimes intended. The finding
is prominent (the simulator would trade at those opens) but ingestion is not refused.
Wired into `ingest_bars` via the dataset's calendar; the 24x7 calendar never flags.

## D33. Benchmarks are a script with a JSON record, not a test — **autonomous**

Plan §9 says not to encode an arbitrary throughput promise. `scripts/benchmark.py` records
wall-clock and peak RSS for a synthetic workload so later optimisation has a baseline.

## D34. Both fill rules run by default — reviewed with the user

Follows from the D14 discussion. `fill_rules` defaults to both rules, so a sensitivity
run is cost scenarios × fill rules. Compute is linear (one more simulation per scenario);
the decisions diverge from the first fill, so the runs cannot share a simulation. The
conservative rule remains the default for single-rule runs and for the first line of any
report. The gap between the twins is pinned by a golden test to
`quantity × (open(N+2) − open(N+1))` per signal fill.

## D35. The textbook rule is the default and the headline — reviewed with the user

Supersedes D14's choice of default (the mechanics it describes are unchanged, and it
remains available as `NEXT_OPEN_AFTER_ELIGIBILITY`).

The user's initial argument was that an academically accepted convention must have a
reason behind it. It does, but the reason does not transfer: the convention comes from
**daily**-frequency research, where hours separate the close from the next open and
"fill at the next open" is unambiguously sound. At intraday frequency the same phrase
makes a weaker claim — the open of bar N+1 prints at `bar_end(N)`, before the bar has
even been received.

The conclusion stands on better grounds. A real fill lands a few seconds into bar N+1;
`open(N+1)` is seconds early, `open(N+2)` is nearly a full bar late. Neither is unbiased
(optimistic overstates fast alpha, conservative erases it), but optimistic is nearer the
truth and its bias has a known direction. Condition attached and kept: the
`optimistic_fill_rule` warning fires on every run using it, and the conservative twin
still runs by default, so the headline number always arrives with its bracket.

## D36. Fixture lateness is 90 s, not 7 minutes — reviewed with the user

The user's principle: do not bake pessimistic assumptions into backtesting, because they
cause more harm than they prevent. The old 7-minute delay in `SyntheticSpec` was invented
to make a code path observable and risked normalising a pessimistic figure.

90 s is the smallest value that exceeds one bar interval, which is what makes availability
run *backwards* (bar N+1 publishes before the late bar N) — the case actually worth
testing; 20 s exercises nothing new. It is also a plausible feed hiccup rather than a
norm. Note that for historical REST ingestion `available_at` is usually a *constant*
offset, under which availability ordering equals interval ordering and D21 is nearly a
no-op; variable lateness matters only with live-recorded or revision-stamped data.

## D37. Two identifiers: `run_id` and `config_id` — reviewed with the user

The user asked why not both, and they were right. `run_id` hashes the resolved spec
*including* `code_revision` (exact provenance, drives reuse and reproduction);
`config_id` hashes it *excluding* `code_revision` and the label, so every run of one
configuration groups together across code edits. A docstring typo no longer scatters a
session's results. Both are stored, indexed, and shown; `runs list --config-id` filters.
Reuse semantics are unchanged, so reproducibility does not weaken.

## D38. Binance label established empirically, not from documentation — **autonomous**

`field[0]` is `bar_start`. Established by pulling BTCUSDT 2024-03-04 12:00Z and rebuilding
the bar from raw `aggTrades` in `[field[0], field[0]+60s)`: open/high/low/close matched
the first/max/min/last trade *exactly*. The kline and the independently derived values are
frozen in `tests/contracts/data/` so the proof runs offline on every test run; a
`network`-marked twin re-pulls and checks the fixture still describes reality.

Reading the documentation would have given the same answer here, but it is not evidence,
and the failure mode (a dataset silently shifted one bar into the future) is invisible in
results. `docs/timestamp_semantics.md` now asks *how* each label was established.

## D39. Publication latency is a collection property, not a venue property — **autonomous**

Measured at two minute boundaries: a closed kline is served 1.1–1.4 s after close, and the
RTT was 1.1–1.4 s — the entire delay is the round trip. Binance publishes closed klines
essentially instantly.

So `publication_latency` for a REST puller is *your* poll interval plus RTT, not something
the adapter can know. It stays a required policy value, the example config says so at
length, and the honest alternative — a live recorder stamping real receipt times into
`available_at` — is named. The adapter never invents a latency.

## D40. Only closed bars are ingested — **autonomous**

A kline whose interval has not ended carries partial OHLCV, and ingesting one writes a bar
that later changes — breaking dataset immutability and back-dating a value that did not
exist. The adapter drops any kline with `close_time >= now` and records the count; if
nothing closed, it raises rather than writing an empty dataset.

Also cross-checks the venue's own `field[6]` against the derived `bar_end` on every
ingest: a mismatch means the interval is not what the adapter assumes, and it refuses
rather than writing a dataset that may be shifted in time.

## D41. Network tests are deselected by default — **autonomous**

The suite must be hermetic and CI must not depend on a third party being reachable.
Live-venue tests carry a `network` marker and skip unless `QRESEARCH_NETWORK_TESTS=1`.
The offline fixture carries the contract; the network test is a freshness check.

## D42. Differential test against an independent simulator — reviewed with the user

Every other test checks qresearch against qresearch's own idea of correct. This one runs
the same strategy on the same bars through `backtesting.py` -- a widely used library with
its own engine -- and requires the fills to agree.

`backtesting.py`'s fill semantics were established the same way the Binance label was:
empirically, with distinct per-bar prices, not from its documentation. An order placed
while bar N is the latest fills at the **open of bar N+1**, which is exactly
`FillRule.OPEN_OF_CURRENT_BAR`. So the test doubles as independent evidence for D35.

Result on a day of real BTCUSDT 1m data: **718 fills on both sides, identical sides and
quantities, fill prices identical to zero difference, final equity identical
(9.3e-16 relative -- float64 rounding).** The conservative rule, run on the same data,
lands 0.227% lower; that gap is ~85% of the strategy's apparent return, which is the
bracket argument made concrete against an outside reference.

One expected difference, asserted rather than papered over: we stamp a fill at
`eligible_at`, `backtesting.py` stamps it at the start of the bar whose open was used.
The offset is exactly the publication latency and the prices are identical, so the equity
comparison replays our fills under their marking convention instead of comparing snapshot
timelines that mean different things.

The test is permanent, marked `oracle`, skipped unless the optional extra is installed,
and runs in CI. Its value is catching engine drift that our own tests would agree with.

## D43. Trades are flat-to-flat, not FIFO lots — reviewed with the user

A trade is one flat-to-flat episode per instrument: it opens when the position leaves
zero, absorbs every add and trim, and closes when it returns to zero. A flip closes one
trade and opens another at the same fill, splitting that fill's costs between them.

Flat-to-flat was chosen over FIFO because the portfolio keeps an *average* cost basis, so
flat-to-flat cannot disagree with the accounting it is derived from and needs no matching
rules to argue about. The cost is that a long accumulation followed by a partial exit is
one trade rather than several — which is the honest description of what the portfolio did.

Validated by an identity asserted on real data across all eight fold/roles: **within each
fold and role, the equity change equals the sum of trade net P&L to 1.15e-15**, with open
trades marked. Friction reconciles to exactly zero difference against the fill ledger.

Note that a trade's `net_pnl` is *not* the engine's `realized_pnl`: the former measures
against the untouched reference price and subtracts every cost including fees, the latter
measures against fill prices and keeps fees on their own ledger line. Both are correct;
they answer different questions.

## D44. An unvalued open trade reports null P&L, not zero and not the cash outflow — **autonomous**

Caught while testing: an open position with no mark was reporting its accumulated cash
outflow (−1000 on a 10-unit buy at 100), which reads as a total loss. Zero would have been
a guess. An unpriced position has *unknown* P&L, so `gross_pnl`, `net_pnl`,
`return_on_notional`, and `exit_price` are null while `costs` — which is known — is kept.

Relatedly, `profit_factor` is null rather than infinite when a run has no losing trades:
an infinity in a headline metric is worse than an absence.

## D45. The UI is a self-contained HTML report per run — reviewed with the user

Chosen over a served dashboard and over notebook helpers. A report written into the run
directory as `report.html` fits the append-only artifact model exactly: versioned with the
run, reproducible from it, openable years later with no server, no CDN and no pinned
plotting library. Charts are hand-rolled inline SVG for the same reason -- adding
matplotlib or plotly would make the artifact depend on a runtime it cannot carry.

**Layout is load-bearing.** Assumptions (dataset, cost scenario, fill rule, latency,
slippage, code revision) come first, warnings sit above the metrics, and only then the
headline tiles. A Sharpe read without its caveats is how people fool themselves, and the
ordering is the cheapest available defence. The footer repeats the pointer to
`docs/leakage_checklist.md`.

Rendering downsamples each series to ~1400 points using min/max-per-bucket rather than
stride sampling, which would drop exactly the spikes a reader is looking for. On a real
day of BTC/ETH that took the file from 388 KB to 140 KB with the shape intact.

Untrusted text (labels, warning messages) is escaped; a test asserts no raw tag can reach
the document and that the tag tree stays balanced, rather than grepping for a payload
substring that appears harmlessly escaped.

## D46. The UI is a layer over the CLI, never a parallel path — reviewed with the user

The report (D45) answered "read one archived run"; it does not answer "explore many runs"
or "author a strategy", which is what a research workbench needs. Both exist now: the
report stays the archival artifact, the app is the exploratory view.

The rule that makes this safe is that **the UI must never create something the CLI cannot
reproduce.** Everything it does resolves to the same `BacktestConfig` -> `RunSpec` ->
`run_id` path that YAML uses, because the platform is already `kind` + `params` over
registries. A UI-only code path would silently void the reproducibility guarantees the
rest of the system is built on. Today the app is strictly read-only and a test asserts the
route table exposes only GET/HEAD; when the launcher lands, that test is what has to be
replaced with an equivalent guarantee, not deleted.

Server-rendered HTML with the same inline-SVG helpers as the report: no build step, no
node toolchain, no JS framework to keep current inside a Python project. The per-run view
serves the archival `report.html` itself, so the browser and the stored file cannot drift.

Binds to loopback with no authentication, deliberately: it exposes a filesystem and will
eventually execute strategy code. On a remote host, forward the port.

## D47. UI and oracle dependencies are optional extras — **autonomous**

`fastapi`/`uvicorn` (`ui`) and `backtesting`/`pandas` (`oracle`) are extras, not runtime
dependencies. The library and CLI must install and work without either. Both are installed
in CI so their tests actually run.

Relatedly, `filterwarnings = error` gained one narrowly-scoped exemption: starlette's
TestClient trips an `anyio` alias deprecation on import. It is matched by message so our
own deprecations still fail the suite — a blanket ignore would have been the easy wrong
answer.

## D48. Jobs are CLI subprocesses, one at a time — reviewed with the user

The runner spawns `python -m qresearch.cli` and supervises it from a thread. The thread
only waits on pipe reads, so the GIL never contends with the engine, and a crash or a
300 MB peak stays in its own process.

Progress needed no new instrumentation: `--verbose --json-logs` already emits `run
started` / `fold complete` (fold, role, fills, return) / `run complete`, each tagged with
`run_id`. The runner tails that stream. Had this not existed, the honest alternative was a
callback interface on the engine that nothing else wanted.

Concurrency defaults to 1. A sweep submitting fifty jobs must not try to run fifty
processes against a benchmark that already peaks near 300 MB.

A job record keeps the **exact argv**, shown in the UI and by `jobs show`, because "what
did this actually run, and could I run it myself?" is the question the whole design exists
to answer. Jobs live under `runs/.jobs/<id>/` beside the runs they produce, so a job, its
config, and its results archive or delete together.

## D49. The GET/HEAD-only test was replaced, not deleted — reviewed with the user

`docs/workbench_plan.md` flagged this as the failure mode to watch: the UI's read-only
guarantee was asserted by checking the route table exposed only GET/HEAD, and job
cancellation is a DELETE.

The replacement asserts the guarantee that actually matters — **not "no mutation" but "no
mutation of results"**: browsing cannot change the set of runs or any run's economic
digest, and every non-GET route must be job control (currently exactly
`DELETE /api/jobs/{job_id}`, asserted by equality so a new one fails the test until it is
justified). Cancelling destroys nothing; it stops work that has not finished.

## D50. Fixed two faults the gates caught — **autonomous**

`filterwarnings = error` and the test suite each caught a real defect rather than noise:

* The runner iterated `process.stdout` without closing it, leaking a file descriptor per
  job — invisible in a test run, material in a long-lived server. Fixed by
  context-managing `Popen`.
* FastAPI's `@app.on_event("shutdown")` is deprecated. That was *our* deprecation, not a
  third party's, so it was fixed by moving to the lifespan API rather than adding an
  exemption. The one standing exemption (starlette's anyio alias) remains the only one.

## D51. Forms are generated from the registries, not hand-written — reviewed with the user

`qresearch/introspect.py` reads each registered implementation's dataclass fields — types,
`Literal` choices, defaults, required-ness — and the launcher renders controls from that.
**Adding a feature in code makes it appear in the UI with no UI change**, which is the same
property that stops the YAML path and the browser path from drifting.

A parameter whose annotation is not recognised is reported as `unsupported` rather than
guessed at, so it surfaces as a gap instead of silently rendering the wrong control. A test
asserts no registered parameter is currently unsupported.

## D52. Launch validates by *constructing* the components — **autonomous**

Pydantic checks shape, not vocabulary: `{"kind": "nope"}` is a perfectly valid
`StrategyRef`, and the registry lookup would not fail until the run was minutes underway.
`_parse` now builds every feature, transform, and strategy before anything is queued, so an
unknown kind, an unknown parameter, or an out-of-range value (`lag: 0`) becomes a
field-level error at submit time.

Discovered by testing the failure path rather than the happy one — the happy path passed
from the first attempt.

## D53. `DEFAULT_CONFIG` holds ISO-8601 strings, not timedeltas — **autonomous**

The defaults are serialised straight into form fields and posted back, so they must already
be in the wire format the parser accepts. A `timedelta` renders as `"6:00:00"`, which does
not parse as a duration — the prefilled form would have been broken on first load.

## D54. The mutating-route test now lists three routes, each justified — reviewed with the user

It fired twice while building Stage 7 (once for `POST /api/backtests`, once for
`POST /api/preview`) and had to be updated deliberately each time, which is exactly the
intent: a new mutating route cannot appear without someone writing down why it is safe.

Current set: `POST /api/preview` (pure computation, POST only because the body is
complex), `POST /api/backtests` (queues the CLI; creates a job, never a run), and
`DELETE /api/jobs/{id}` (stops unfinished work). A companion test asserts launching does
not change the set of runs.

## D55. UI assets are static files, not Python string literals — **autonomous**

Keeping JS and CSS inside `.py` strings meant fighting the line-length limit with code that
gets no syntax highlighting and no linting. They now live in `qresearch/ui/static/` and are
served by the app; `pyproject.toml` ships them in the wheel. The archival report stays
fully self-contained and keeps its inline styles — different artifact, different rule.

## D56. The UI is tested in a real browser — reviewed with the user

**A syntax error shipped in `launch.js` and every test passed.** The launcher was served,
rendered, and completely inert: one malformed escape meant the whole script failed to
parse, so no dropdown populated and no button did anything. The suite asserted
`client.get("/static/launch.js").status_code == 200`, which proves a file was served, not
that it is valid JavaScript.

`tests/integration/test_ui_browser.py` now drives a real chromium against a real uvicorn
process: pages must load with no JavaScript errors, the launcher must populate itself from
the API, changing a component must re-render its parameters, and preview must draw a fold
ribbon. Verified to fail on the original bug — re-introducing it fails six tests. A
`node --check` test catches the same class faster when node is available.

The extraction of JS from Python string literals (D55) is what corrupted the escape. The
lesson is not "do not extract" — static files are still right — but that *serving is not
working*, and only executing the page proves the difference.

Page errors and failed requests are tracked separately: a 422 from submitting a
deliberately invalid config is the behaviour under test, and folding it into a single "no
errors" assertion would make the negative tests lie.

## D57. Closed another leaked pipe, in the test fixture this time — **autonomous**

The browser test's uvicorn fixture used `Popen(stdout=PIPE)` without closing it, which
pytest surfaced as an unraisable exception at teardown — the same defect the job runner had
(D50). Both are now closed explicitly.

## D58. Stage 8: rules are a condition tree, not a language — reviewed with the user

`RuleStrategy` evaluates comparisons over features combined with all/any/not, producing one
target weight. No loops, no arithmetic, no state. Anything it cannot express is a
`Strategy` class, and that path already works — which is the guard against this growing
into a DSL nobody can test.

Rules are *data*: serialisable, part of the `RunSpec`, hashed into the `run_id`. A rule
change is a new run, the rule is stored with its results, and there is no sandbox question
because nothing is executed.

A comparison whose feature is missing or null evaluates to **false, not an error**.
Warm-up leaves most features null and a rule that threw there would make the first bars of
every run unusable. The cost is that a mistyped feature name reads as "never true" rather
than failing loudly, so `Rule.referenced_features` exists to check against the configured
feature set. Long takes precedence over short when a rule somehow says both — stated
rather than emergent.

`max_positions` breaks ties by `rank_by` then instrument id, so selection is deterministic
rather than dependent on dictionary order.

## D59. Cross-sectional ranks are now part of a run — **autonomous**

`CrossSectionalRank` existed and was tested but the orchestrator never called it, so no
strategy could select across instruments. `BacktestConfig`/`RunSpec` gained
`cross_sectional`, applied after the per-instrument features, and a rank on a column no
feature produces is refused at config time rather than silently reading as "never true".

## D60. `fetch_instruments` reads the venue's own exchangeInfo — **autonomous**

Tick size, step size and base/quote are facts Binance publishes; hand-writing them is
tedious and a wrong `quantity_increment` silently changes every order size in a backtest.
It also surfaces status: fetching the 10-name universe flagged MATICUSDT as not TRADING.
`listed_from` defaults to before Binance existed, which is right for a fixed universe and
wrong for a point-in-time one — stated at the call site.

## D61. A strategy reading a feature the run does not produce is refused — **autonomous**

D58 chose to make a missing feature evaluate to **false** rather than raise, so warm-up
would not make the first bars of every run unusable, and noted the cost: a mistyped name
reads as "never true". That cost arrived immediately. A parameter sweep varied the z-score
window, which renamed the feature from `zscore_30` to `zscore_72`, while the rule kept
referencing the old name. **All six trials completed successfully with zero trades** —
indistinguishable from a strategy that found no opportunities.

Strategies may now advertise `required_features`, and the orchestrator checks them against
the features the run will produce (including `_xrank` columns) *before* loading any data.
`RuleStrategy` reports its rule's references plus `rank_by`. The launcher had this check;
runs did not, which is the gap that mattered.

The general lesson: a design that trades loudness for convenience needs the loudness put
back somewhere specific, and "somewhere" must include every entry point, not the one being
worked on at the time.

## D62. `derive_dataset` makes coarser bars a first-class dataset — **autonomous**

`resample_bars` existed but had no path to a stored dataset, so testing a strategy at 5m
meant hand-rolling the pipeline. `qresearch data resample <id> --to 5m` writes a real
dataset with its own id whose source records the parent id and parent content digest, so
lineage is explicit and re-deriving resolves to the same id.

## D63. Sweeps compare on validation and print the trial count — **autonomous**

`scripts/sweep.py` expands a grid, runs each point under one `experiment_id`, and sorts by
**validation** return with test shown beside it. It prints "best of N trials" with the
winner, because the maximum of many draws is biased upward and the leakage checklist makes
multiple comparisons a reporting requirement rather than a footnote.

`linked` grid keys apply several paths together, for parameters that must move in step: a
symmetric entry threshold varied independently would generate long −2 / short +3 pairs
nobody intended to test.

## D64. Result: the signal is real and does not clear costs — reviewed with the user — *friction units corrected in D67*

Six months of 5-minute Binance bars (10 instruments, 524,160 rows, Jan-Jun 2024),
11 walk-forward folds. Strategy: long the most-volatile names when stretched below -3σ of
their 6-hour mean, short when above; at most two positions.

| | validation | test | trades | cost drag |
|---|---|---|---|---|
| 3σ, no costs | +9.50% | **+11.43%** (Sharpe 2.75) | 866 | — |
| 3σ, base costs | −3.60% | −3.10% | 866 | 13.9% |
| 2σ, no costs | +24.59% | −3.31% | 2,644 | — |
| 2σ, base costs | −24.00% | −41.11% | 2,662 | 48.5% |

The 3σ signal is **persistent out of sample**: positive in 9 of 11 test folds, 66.9% win
rate, profit factor 1.32, across a rally and a drawdown. This is not the five-day result
repeated — it survived a twenty-fold increase in sample.

It also does not clear costs, and the margin is specific: **$12.64 gross per trade against
$16.05 of modelled friction**. At the assumed 2.5 bps half-spread that leaves roughly
2 bps of affordable round-trip friction, about half what the model charges.

The 2σ variant is the counter-example that makes the point: it had the best validation
return in the five-day sweep and posts −3.31% gross on test over six months. More trading
bought noise, not edge.

What this does *not* establish, and the report says so on every run: the optimistic fill
rule is in use (the conservative twin would be worse); shorts carry no borrow cost and the
strategy opens thousands of them; the universe is ten names chosen today, so survivorship
is unmodelled; and the 2.5 bps half-spread is a flat assumption applied to both BTC and
DOGE, when the strategy concentrates in the wider-spread names. Settling whether ~2 bps is
achievable needs quote data, which this platform does not yet ingest.

## D65. Metrics annualise over the decision range, not the whole simulation — **autonomous**

**Every annualised number reported before this was understated.** A walk-forward fold's
equity curve covers warm-up, training and purge as well as the evaluated window, and
`compute_metrics` annualised over all of it. On the six-month study that was 176 days of
curve for 77 days of trading, so every rate was divided by 2.3x too many periods:
annualised return, volatility, Sharpe and turnover alike.

Found because a hand calculation of the annualised return disagreed with the reported
figure, and the ratio between them was exactly the ratio of span to decision time. Worth
noting the failure mode: the number was not absurd, just wrong, and nothing in the suite
compared it against an independent calculation.

`compute_metrics` now takes an optional `decision_range` and the orchestrator supplies it,
both per fold and when stitching the per-role aggregate. Corrected figures for the 3σ
survivor on test: annualised return −13.80% (was −6.34%), turnover 1,619x/year (was 707x),
Sharpe −0.82 (was −0.97); cost-free +67.08% annualised at Sharpe 3.18.

Trimming can leave a fold empty -- an equity fold whose test window falls outside trading
sessions has no snapshots -- so stitching drops empty curves rather than asking an empty
column for its minimum.

## D66. The spread was measured; the fee is published, and it is the binding constraint — reviewed with the user — *spread figure corrected in D67*

Two cost inputs were wrong in opposite directions, and the net effect is worse than the
base scenario suggested.

**Spread: measured, and the assumption was 6x too pessimistic.** Binance publishes no
historical quotes for spot (`bookTicker` exists only for futures), but `aggTrades` carries
`isBuyerMaker`, so an effective half-spread can be recovered by comparing mean ask-side and
bid-side prices within 5-second windows. On 2024-03-04: BTCUSDT **0.043 bps**, SOLUSDT
0.361, DOGEUSDT 0.422 — against the 2.5 bps the base scenario assumes.

**Commission: not a measurement, and it dominates.** Binance spot retail is 0.1% per side
(10 bps), ~0.075% with BNB, falling to low single-digit bps only at high VIP tiers. The
model assumed 1.0 bps — roughly ten times too generous. This cannot be measured from public
data; it is a fee tier that has to be confirmed for the account that would trade.

Six months, test, 3σ survivor, measured 0.4 bps half-spread:

| commission | test return | annualised | Sharpe | cost drag |
|---|---|---|---|---|
| 10 bps (retail) | −23.16% | −71.3% | −7.50 | 36.6% |
| 2 bps (high VIP) | +0.57% | +2.7% | 0.25 | 10.2% |
| 0 bps (maker/rebate) | +7.55% | +41.3% | 2.17 | 3.5% |

The strategy is not spread-constrained, as the base scenario implied — it is
**fee-constrained**, and at retail rates it is not close. `measured_retail`,
`measured_vip` and `measured_maker` are now named scenarios so this is reproducible.

Caveat that stops this being a recommendation: the maker row assumes passive fills, which
this platform cannot simulate — bar data cannot model queue position, and a limit order
that earns the spread is also an order that does not always fill. Testing it needs the
quote/trade execution model in ARCHITECTURE.md sec. 11, not another cost parameter.

## D67. Cost inputs audited: units, spread source, venues, fees — reviewed with the user

Asked what service the cost estimate came from and whether the alternatives had been
checked. The honest answer was one venue's public archive, one day, three symbols, and a
fee quoted from memory. This entry replaces that.

**Units (corrects D64).** Every bps figure the engine and reports produce is per unit of
*traded* notional — per side. A round trip trades a position twice (2.02x entry notional
on this strategy), so round-trip figures are double. D64's "roughly 2 bps of affordable
round-trip friction" was wrong: the 3σ survivor's gross edge on test is **3.25 bps per
side (~6.6 round trip)**, against **4.15 bps per side (~8.3) of base-scenario friction**:
2.50 spread, 0.65 impact, 1.00 commission.

**Framework.** Cost per fill = half-spread + size impact + commission, each in bps of fill
notional; a strategy breaks even when gross edge per side equals cost per side. Only the
spread is measured from data. Impact is a model — 25 bps × quantity / trailing bar volume;
median participation 1.6%, and 27% of test fills already hit the 5% cap at $100k.
Commission is a published tariff that depends on venue, region and volume.

**Spread: source and method.** `data.binance.vision`, Binance's public archive. Spot has no
quote history, so the half-spread is inferred from `aggTrades`: per 5-second window, mean
buyer-initiated price minus mean seller-initiated price, over the mean price, halved;
median of positive windows.

- *Calibrated* where quotes exist: DOGEUSDT perps 2024-03-12, inferred 0.293 vs quoted
  0.295 bps (0.99x). Perp quote archives stop after 2024-03-30, so the fill days cannot be
  checked the same way.
- *Validated* independently: on 2024-04-28 nine of ten spot names measured exactly half a
  price tick (AVAX 1.430 vs 1.436, ADA 1.061 vs 1.063, XRP 0.961 vs 0.962, ...). Those books
  were one tick wide, so their spread is set by tick size, not sampled.
- *Conditioned* on the strategy: the 5s windows around its fills on each name's three
  busiest test days, weighted by traded notional.

| name | notional share | spot at fills | perp at fills |
|---|---|---|---|
| DOGE | 20.3% | 0.60 | 0.48 |
| SOL | 17.5% | 0.42 | 0.34 |
| AVAX | 17.3% | 1.46 | 0.32 |
| ADA | 9.3% | 0.98 | 0.83 |
| LINK | 8.8% | 0.35 | 0.28 |
| BNB | 7.6% | 0.77 | 0.21 |
| XRP | 7.1% | 0.80 | 0.81 |
| ETH | 5.3% | 0.08 | 0.21 |
| MATIC | 4.6% | 0.56 | 0.59 |
| BTC | 2.3% | 0.14 | 0.23 |
| **weighted** | | **0.72** | **0.43** |

Perp spreads widen 1.2–1.9x around fills; a tick-bound spot book cannot widen. D66's 0.4
bps was extrapolated from BTC, SOL and DOGE and missed AVAX, ADA, BNB, XRP and MATIC. The
`measured_*` scenarios now use 0.72. Rerun, six months, test, 3σ survivor:

| scenario | commission | test return | annualised | Sharpe | cost drag | D66 at 0.4 bps |
|---|---|---|---|---|---|---|
| `measured_retail` | 10 bps | −23.98% | −72.8% | −7.81 | 37.7% | −23.16% |
| `measured_vip` | 2 bps | **−0.51%** | −2.4% | −0.06 | 11.3% | +0.57% |
| `measured_maker` | 0 bps | +6.40% | +34.2% | 1.86 | 4.6% | +7.55% |

The 2 bps row turns negative. `measured_maker` is misnamed: it still *pays* the half-spread,
so it is a zero-fee taker, not a maker — a real passive fill would earn it.

**Fees.** Read from venue pages on 2026-09-12. Taker bps per side; "at ~$13M" is the tier
this strategy's own volume reaches at $100k (4.35x equity per day).

| venue | product | base | at ~$13M / 30d | best published | source |
|---|---|---|---|---|---|
| Binance | spot | 10 (7.5 in BNB) | not captured | 2.3 | binance.com/en/fee/schedule |
| OKX (global) | spot | 10 | 6.5 (notice) | 1.5 (live) / 1.75 (notice) | okx.com/fees; fee framework notice, Nov 2025 — they disagree |
| Bybit | spot | 10 | not captured | 4.5 | bybit help centre |
| Kraken | spot | 80 | not captured | 5 | kraken.com/features/fee-schedule |
| Coinbase Advanced | spot | 60 UNVERIFIED | — | 4 UNVERIFIED | pages returned 403 |
| Binance.US | spot, Tier 0 pairs | 1 | 1 | 1 | binance.us/fees; pair list not in page |
| Kraken Futures | perp | 5 | 4.0 | 1.25 | futures.kraken.com fee API |
| OKX (global) | perp | 5 | 4.0 | 1.5 | okx.com fee framework, Nov 2025 |
| Bybit | perp | 5.5 | not captured | 3.0 | bybit help centre |
| Binance | USDⓈ-M perp | could not verify | | | table renders client-side |

Region decides the list before fees do: Bybit and Kraken Futures exclude US persons, and
Binance.com lists the US as restricted (UNVERIFIED). OKX quotes different base fees by
region, from 8/10 to 70/70.

**Spread across venues.** Quoted top of book, 10 polls over one minute on 2026-09-12,
weighted by the strategy's mix: Binance spot 0.64, Binance perp 0.64, Bybit perp 0.69,
OKX perp 0.70, Bybit spot 0.74, OKX spot 0.79, Kraken spot 0.81, Coinbase 1.01, Kraken
Futures 1.11, Binance.US 2.34 (USDT pairs) / 5.41 (USD pairs). On the liquid names the
major venues quote the same spread; **the venue choice moves the fee, not the spread**.
Today's prices and ticks over one minute — this ranks venues, it does not re-price 2024.

**Conclusion.** Taker, per side: gross 3.25 bps less spread and modelled impact leaves
**~2.2 bps of fee headroom on perps and ~1.9 on spot**. The cheapest route found at the
strategy's own volume is perps at 4.0 bps, costing ~5.1 per side and losing ~1.8. Taker
fees near 2 bps need ≥$100M per 30 days (Kraken Futures), 7.7x today's capital, while a
quarter of fills are already participation-capped at $100k: the strategy cannot trade its
way into a tier that rescues it. Binance.US Tier 0 is the only sub-2 bps taker fee without a
volume requirement, and its books are 4–8x wider on this mix with depth unmeasured.

Not settled here: maker execution (needs the quote-level model in ARCHITECTURE.md sec.
11); executing a spot-bar signal on perps (basis and funding — test by ingesting perp
klines, which the Binance adapter does not yet support); and impact, which is modelled
rather than measured.

## D68. Research capital is $10k; July–December 2024 is sealed as the holdout — capital at the user's request; the holdout is **autonomous**

The user will not start with $100k. The crypto example configs, `my_first_backtest.yaml`
and the UI launch form now start at $10,000; the engine's library default and the equities
example are unchanged. Scale leaves returns, drawdowns and bps as they were. What it changes
is the fee tier: a $10k account trades about $1.3M a month and pays base rates, ~10 bps per
side on spot and ~5 on perps (D67). A strategy therefore needs roughly **22 bps gross per
round trip on spot and 11 on perps** before it breaks even.

Jan–Jun 2024 has been evaluated over and over while the strategy, its threshold and its
cost model were chosen, test folds included. Its test folds can no longer serve as
out-of-sample evidence, so that period is now a development set and anything may be looked
at. **July–December 2024 is sealed**: it is ingested, and nothing is computed on it (no
features, event study or backtest) until a candidate and its parameters are fixed. It is
then run once, and the result is reported whatever it is. MATICUSDT is left out because
Binance migrated MATIC to POL during that period.
