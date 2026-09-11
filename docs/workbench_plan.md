# Workbench plan (Stages 6–9)

`DEVELOPMENT_PLAN.md` covered the MVP: ingest, features, simulate, evaluate, CLI. It is
complete. This plan covers the next goal — **a UI you can actually work in**: choose data,
author a strategy, launch a backtest, and read the results without touching YAML.

Stage 6 landed already in read-only form (the runs explorer). What follows turns it from a
viewer into a workbench.

## The organising decision

**The UI drives the CLI as a subprocess. It does not call the engine in-process.**

The UI writes a config, spawns `qresearch backtest run -c <config> --json-logs`, and
streams the result. That is not a shortcut — it is what enforces the rule in D46, that the
UI can never create something the CLI cannot reproduce. If the browser's only power is to
write a config and run the same command you would, reproducibility is structural rather
than something to remember.

It is also already supported. Measured on this repo:

* CLI startup is ~0.5 s, negligible against a run.
* `--verbose --json-logs` already emits `run started` → `fold complete` (with `fold`,
  `role`, `fills`, `return`) → `run complete`, every line tagged with `run_id`. **A UI
  needs no new instrumentation to show progress.**
* A separate process gives isolation for free: a crash or a 300 MB peak does not touch the
  server, and cancelling is a kill.

Side benefit: the config the UI writes *is* an artifact. Copy it, run it yourself, get the
same `run_id`.

---

## Stage 6 — Job runner

**Goal:** submit → progress → done, surviving a page refresh.

Deliverables:

- `JobSpec` / `JobRecord` contracts: id, kind (`backtest` / `ingest`), config path, state
  (`queued`/`running`/`succeeded`/`failed`/`cancelled`), timestamps, produced `run_id`s,
  log path, exit code.
- A job store under `runs/.jobs/<job_id>/` with `config.yaml`, `job.json`, `log.jsonl` —
  atomic writes, same append-only pattern as run artifacts.
- A runner that spawns the CLI, tails stdout/stderr into `log.jsonl`, parses the structured
  lines into progress, and records the outcome.
- A bounded queue (default: 1 concurrent job) so a sweep cannot fork-bomb the machine.
- UI: a Jobs page, per-job progress (fold *n* of *m*), cancel, and a link to results.
- `GET /api/jobs`, `POST /api/jobs`, `DELETE /api/jobs/{id}` (cancel).

Acceptance criteria:

- A submitted job survives a server restart in a terminal state; an interrupted one is
  recorded as `failed`, never silently lost.
- Cancelling kills the process and leaves no partial run in the store (the store's atomic
  `.tmp` → publish already guarantees this; the test must prove it).
- The config written for a job, run by hand through the CLI, produces the same `run_id`.
- Two concurrent submissions queue rather than running at once.

**Risk:** orphaned processes if the server dies mid-job. Mitigation: record the PID, reap
on startup, mark unknown-state jobs `failed`.

---

## Stage 7 — Backtest launcher

**Goal:** configure and launch a run from the browser.

Deliverables:

- Dataset picker from the catalog, with instrument multi-select.
- Feature picker generated from `FEATURES` — parameter forms derived from each
  implementation's dataclass annotations, which `features.registry.construct` already
  reads. New feature in code → form in the UI, no UI change.
- Strategy picker from `STRATEGIES`, same mechanism.
- Cost scenario and fill-rule selection, defaulting to the bracket.
- **Fold-layout preview before running**: `generate_folds` is pure, so the ribbon can be
  drawn from the plan without executing anything. Catching a wrong purge or a fold that
  does not fit costs a second here instead of minutes later.
- "Show config" — the exact YAML, copyable.

Acceptance criteria:

- Every form maps onto `BacktestConfig`; invalid combinations are rejected by the same
  Pydantic validation the YAML path uses, with the same messages.
- The preview's folds match what the run actually uses.
- A run launched from the UI is byte-identical in `run_spec.json` to the same config run
  from the CLI.
- **The D46 guarantee is preserved under mutation.** The current test asserts the route
  table is GET/HEAD-only; it must be *replaced* by one asserting every mutating route
  produces a config that round-trips through the CLI, and that no route writes to the run
  store except via the subprocess. Deleting that test without a replacement is the failure
  mode to watch for.

---

## Stage 8 — Rule builder

**Goal:** author a strategy without writing code, without inventing a language.

`lagged_signal(feature, weight, threshold)` is a demo. The gap is real: three templates is
not a research tool.

Deliverables:

- `RuleStrategy`: a serialisable condition tree evaluated per instrument per decision.
  Deliberately small — comparisons between a feature and a constant or another feature,
  combined with AND/OR/NOT, producing a target weight. Explicitly **not** a DSL: no loops,
  no arbitrary arithmetic, no state.
- The rule is *data*, part of `RunSpec`, hashed into `run_id` — so a rule change is a new
  run, and the rule is stored with its results.
- Cross-sectional conditions via the existing `CrossSectionalRank` ("top 20 % by `ret_1`"),
  reusing the batch-availability semantics already built and tested.
- UI: condition rows with feature/operator/value dropdowns, and a signal preview over a
  chosen day before launching.
- The escape hatch stays: write a `Strategy` class, register it, it appears in the UI with
  its parameters as a form.

Acceptance criteria:

- `RuleStrategy` passes `assert_prefix_invariant` and `assert_future_insensitive` on
  fixtures with gaps and late bars, like every feature.
- A rule round-trips through JSON and reproduces identical fills.
- An equivalent rule and hand-written strategy produce identical ledgers.
- Rules referencing an unknown feature fail at config time, not mid-run.

**Risk:** scope creep into a general expression language. The guard is that anything the
builder cannot express is a Python class, and that path already works.

---

## Stage 9 — Data screen and sweeps

**Goal:** get data and explore parameter space without leaving the UI.

Deliverables:

- Ingest form: venue, symbols, interval, date range → a `binance://` URI and an ingest job.
- **Auto-populate instrument definitions** from Binance `/api/v3/exchangeInfo` (tick size,
  step size, base/quote) — this removes the most tedious part of the YAML today.
- Dataset detail: coverage, validation findings, gaps, point-in-time preview.
- Sweep: mark parameters as ranges, expand to a grid, submit as queued jobs, show results
  as a surface or table with one click through to any run.

Acceptance criteria:

- An ingest launched from the UI produces the same `dataset_id` as the equivalent YAML.
- Fetched instrument definitions match a hand-written config for the same symbols.
- A sweep of *n* points produces *n* runs sharing an `experiment_id`, each independently
  reproducible.
- The sweep view shows the trial count alongside the best result (the multiple-comparisons
  warning in `docs/leakage_checklist.md` is a reporting requirement, not a footnote).

---

## Explicit non-goals

- **In-browser code editing.** The registry path covers it; an editor adds an execution and
  reload problem for little gain over the editor already open.
- **Multi-user, auth, hosting.** Loopback, single user. Serving this publicly is not a
  supported configuration — it exposes a filesystem and runs strategy code.
- **Live trading or order routing.** Unchanged from `ARCHITECTURE.md` §12.
- **Replacing the CLI.** The UI is a front end to it, permanently.

## Sequencing note

Stage 6 is the prerequisite for 7 and 9, and 8 is independent of all three — the rule
builder can be built and tested headlessly against the existing engine before it has any
UI at all. If the job runner proves fiddly, Stage 8 is the thing to do in parallel.
