# Execution assumptions

What the simulator assumes about getting filled, and what it refuses to assume. Every
item here is a configuration value or a fixed rule with a test; none is a hidden default.

## What the execution model may see

Only the **opening print** of the bar whose open event is being processed, plus two
numbers attached to the order when it was created: a trailing-volume estimate and a
trailing-volatility estimate computed from bars **published by `order_at`**. It never
sees that bar's high, low, close, or volume — they do not exist yet — and it never calls
strategy code.

## When an order can fill

Default: `FillRule.NEXT_OPEN_AFTER_ELIGIBILITY`. An order fills at the first bar-open
event at or after its `eligible_at`, processed in a later clock step than the decision
that created it.

Consequence worth internalising: with contiguous bars, bar N is published no earlier than
`bar_end(N) == bar_start(N+1)`, so bar N+1's open has already printed when a decision on
bar N is made. **A signal on bar N fills at open(N+2)** under any non-negative latency.
This is one bar more conservative than the textbook "next open" assumption.

Optional: `FillRule.OPEN_OF_CURRENT_BAR` fills at `eligible_at` at the open price of the
bar containing it — the textbook assumption. It uses a print from before the order
existed and is optimistic by up to a bar of latency; every run using it records an
`optimistic_fill_rule` warning. It exists because 5-minute research is materially
distorted by a 5-minute delay.

Latencies: `submission_latency` (signal → order) and `order_latency` (order → eligible),
both configurable, default 0 s and 1 s. Every fill asserts
`signal_at <= order_at <= eligible_at <= fill_at`.

## Price

```
price = open + side * (half_spread + slippage)
fee   = max(quantity * price * commission_bps/1e4 + quantity * fee_per_unit, min_commission)
```

Each component is stored on the fill. Defaults are non-zero (2.5 bps half-spread, 1 bps
commission, participation slippage at 25 bps per 100 % participation). Spread is an
assumption when no quote data exists and is labelled as such.

Slippage kinds:

| kind | formula | needs |
|---|---|---|
| `fixed_bps` | `open * coef/1e4` | — |
| `participation` | `open * coef/1e4 * quantity/liquidity` | liquidity estimate |
| `sqrt_impact` | `open * coef * volatility * sqrt(quantity/liquidity)` | both estimates |

When an estimate is missing (fewer than `liquidity_lookback_bars` published), slippage
falls back to `fallback_bps` and a `slippage_fallback` warning is recorded.

## Size

`participation_cap` (default 10 %) limits each fill to a fraction of the liquidity
estimate; the remainder stays pending and fills at later opens (partial fills are
`OrderEvent`s). With no estimate the cap cannot apply: the order fills uncapped and
`no_liquidity_estimate` is recorded.

Quantities are rounded toward zero to the instrument's `quantity_increment`.

## Waiting, expiry, gaps

Unfilled orders expire `expire_after` after `order_at` (default 5 min). A missing next bar
never produces a fill at a stale price; the order waits for the next real open event or
expires (`MissingBarPolicy.WAIT`). Expiry coinciding with an open expires first.

## Marks and valuation

Positions are marked at the latest **published** close. A position filled before its
instrument has ever published is marked at the print it traded on until a close arrives
(first-bar edge case only). A position valued at a mark older than `max_mark_staleness`
at a decision records `stale_mark`. Nothing is valued during gaps where no decision is
made, so a single instrument's gap produces no warning; a multi-instrument gap does.

## End of run

`EndOfRunPolicy.MARK` (default): positions stay open at the last point-in-time close.
`LIQUIDATE`: at the first instant at or after the decision range end, pending orders are
cancelled and flattening orders submitted; opens keep being processed until they fill or
expire. If that instant is also an open event, in-range orders that can fill do so first.
No data past the end → `liquidation_impossible`.

## Constraints

Applied to the whole intent batch at once. Shorting off by default; per-name and gross
exposure limits; cash check with a cost buffer. Cross-instrument limits scale every
exposure-increasing order by one common factor, so results do not depend on intent
order. Targets net against current position **plus pending orders**. Every resize and
rejection is an `IntentOutcome` row.

## Not modelled

Limit or stop orders; intrabar paths; queue position; borrow cost or availability for
shorts (`short_position` warning); funding; cross-currency cash; corporate actions in
execution prices. See ARCHITECTURE.md §12.
