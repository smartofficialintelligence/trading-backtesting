# Leakage and realism checklist

Read this against every result before believing it. Each item names the control that
exists in the code and the residual risk that a person still has to judge.

## Enforced by construction

- [ ] **Bar label convention declared** — `timestamp_label` is a required field; the
      two conventions produce different dataset ids. *Residual:* the declaration can be
      wrong. Verify against a known market event (docs/timestamp_semantics.md).
- [ ] **Availability filter on every read** — `BarQuery.as_of` is mandatory; unfiltered
      reads require a stated reason. `grep scan_all_unfiltered` in research code.
- [ ] **No same-bar or next-open fills** — the engine fills a signal on bar N at open(N+2)
      by default. If `fill_rule: open_of_current_bar` appears in the run, the
      `optimistic_fill_rule` warning is set; treat results as an upper bound.
- [ ] **Order timestamps distinct** — every fill asserts
      `signal_at <= order_at <= eligible_at <= fill_at`.
- [ ] **Feature availability derived, not written** — window max of input availability
      plus latency; cross-sectional ranks lift to the batch. New features must pass
      `assert_prefix_invariant` and `assert_future_insensitive` on fixtures with gaps
      and late bars.
- [ ] **Transforms fitted per fold on train rows only** — `fit` accepts `TrainingData`
      only. Check `fitted/` in the run directory: one state per fold, `fitted_on` inside
      the fold's training range.
- [ ] **Purge stated** — `label_horizon` or `purge` is required; check it is at least
      the longest horizon any label or feature looks forward.
- [ ] **Costs non-zero** — compare `base` against `free` and `stressed`; a strategy that
      only works under `free` does not work.
- [ ] **Liquidity from the past** — participation caps and impact use estimates from
      bars published by `order_at`; `no_liquidity_estimate` warnings mean uncapped fills.
- [ ] **Accounting reconciled** — `equity == initial + realized + unrealized - fees` is
      asserted at every snapshot.
- [ ] **Deterministic** — identical spec reuses the run; `runs reproduce <id>` re-executes
      and compares the economic digest.

## Judged by a person

- [ ] **Survivorship** — the universe is whatever was ingested. There is no point-in-time
      membership yet; a static list of today's liquid names applied to history is biased.
- [ ] **Corporate actions** — `price_adjustment` is recorded but not applied in
      execution; do not mix adjusted signals with unadjusted fills.
- [ ] **Test peeking** — validation and test metrics are stored separately and labelled;
      the software cannot stop you reading the test column while tuning. Keep a note of
      how many times you looked.
- [ ] **Multiple comparisons** — every trial is a stored run under an `experiment_id`;
      report the trial count with the winner.
- [ ] **Regime coverage** — per-fold metrics are in `metrics.json`; a strategy that wins
      in one fold of four is one fold of four.
- [ ] **Spread assumption** — `half_spread_bps` is an assumption without quote data.
- [ ] **Shorts** — no borrow cost or availability (`short_position` warning).
- [ ] **Stale marks** — `stale_mark` warnings mean a position was valued at an old price.
- [ ] **Session sanity** — `outside_session` findings on an equity dataset mean bars the
      venue was closed for; the simulator would trade at their opens.
