"""Differential test: qresearch against an independent simulator.

Every other test in this suite checks qresearch against qresearch's own idea of what is
correct. This one runs the same strategy on the same bars through ``backtesting.py`` --
a widely used library written by other people, with its own engine -- and requires the
fills to agree.

The comparison is only meaningful once both are forced into identical semantics, so:

* ``backtesting.py`` fills an order placed while bar N is the latest bar at the **open of
  bar N+1**. That was established empirically (not from its documentation) with distinct
  per-bar prices, the same way the Binance label was. It corresponds exactly to
  ``FillRule.OPEN_OF_CURRENT_BAR``, which is why that is the rule under test here.
* Costs, slippage, spread, and latency are zero on both sides; participation caps,
  expiry, and exposure limits are disabled. Any of them would be a legitimate reason to
  differ and would tell us nothing.
* The strategy is deliberately trivial and unambiguous -- hold one unit while the last
  completed bar's return was positive -- so there is no room for the two libraries to
  interpret it differently.

One difference is expected and is *not* a disagreement: qresearch stamps a fill at
``eligible_at`` (the instant the order reached the venue), while ``backtesting.py``
stamps it at the start of the bar whose open was used. The offset is exactly the
publication latency, and the fill *prices* are identical. The equity comparison therefore
replays our fills under their marking convention (position held during bar N, marked at
close N) rather than comparing snapshot timelines that mean different things.

Run with: ``uv sync --extra dev --extra oracle && uv run pytest -m oracle``
"""

from __future__ import annotations

import datetime as dt
from random import Random
from typing import Any

import polars as pl
import pytest

from qresearch.data.contracts import AssetClass, Instrument
from qresearch.features.pipeline import compute_features
from qresearch.features.technical import LaggedReturn
from qresearch.research.splits import TimeRange
from qresearch.simulation.constraints import ConstraintConfig
from qresearch.simulation.engine import EndOfRunPolicy, SimulationConfig, run_simulation
from qresearch.simulation.events import OrderIntent
from qresearch.simulation.execution import CostConfig, ExecutionConfig, FillRule

backtesting = pytest.importorskip("backtesting", reason="needs the 'oracle' extra")
pd = pytest.importorskip("pandas", reason="needs the 'oracle' extra")

pytestmark = pytest.mark.oracle

T0 = dt.datetime(2024, 3, 4, tzinfo=dt.UTC)
SYMBOL = "CRYPTO:BTCUSDT"
BARS = 400
CASH = 1_000_000.0
LATENCY = dt.timedelta(seconds=2)
from decimal import Decimal  # noqa: E402


def bars_frame(count: int = BARS, seed: int = 7) -> pl.DataFrame:
    """A deterministic random walk. Distinct opens and closes, no gaps."""
    rng = Random(seed)
    price, rows = 65_000.0, []
    for i in range(count):
        start = T0 + dt.timedelta(minutes=i)
        open_ = price
        close = round(open_ * (1.0 + rng.gauss(0.0, 0.0008)), 2)
        wick = abs(rng.gauss(0.0, 0.0004)) * open_
        rows.append(
            {
                "instrument_id": SYMBOL,
                "bar_size": "1m",
                "bar_start": start,
                "bar_end": start + dt.timedelta(minutes=1),
                "available_at": start + dt.timedelta(minutes=1) + LATENCY,
                "open": open_,
                "high": round(max(open_, close) + wick, 2),
                "low": round(min(open_, close) - wick, 2),
                "close": close,
                "volume": round(10.0 + rng.random() * 5, 4),
                "vwap": None,
                "trade_count": None,
                "source": "oracle-test",
                "source_key": None,
                "revision": 0,
                "ingested_at": T0,
            }
        )
        price = close
    return pl.DataFrame(
        rows,
        schema_overrides={
            "bar_start": pl.Datetime("us", "UTC"),
            "bar_end": pl.Datetime("us", "UTC"),
            "available_at": pl.Datetime("us", "UTC"),
            "ingested_at": pl.Datetime("us", "UTC"),
            "vwap": pl.Float64,
            "trade_count": pl.Int64,
            "revision": pl.Int32,
        },
    )


class HoldWhileLastReturnPositive:
    """Hold one unit while the last completed bar's return was positive, else flat."""

    strategy_id = "oracle_differential:v1"

    def reset(self) -> None:
        return None

    def on_decision(self, context: Any) -> Any:
        value = context.feature(SYMBOL, "ret_1")
        if value is None:
            return ()
        target = 1.0 if value > 0 else 0.0
        if abs(context.position(SYMBOL) - target) < 1e-12:
            return ()
        return [OrderIntent.target_quantity(SYMBOL, target, at=context.decision_at)]


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return bars_frame()


@pytest.fixture(scope="module")
def ours(bars: pl.DataFrame) -> Any:
    instrument = Instrument(
        instrument_id=SYMBOL,
        asset_class=AssetClass.CRYPTO,
        venue="TEST",
        quote_currency="USDT",
        base_currency="BTC",
        price_increment=Decimal("0.01"),
        quantity_increment=Decimal("0.00001"),
        calendar_id="24x7:1",
    )
    config = SimulationConfig(
        initial_cash=CASH,
        execution=ExecutionConfig(
            submission_latency=dt.timedelta(0),
            order_latency=dt.timedelta(0),
            fill_rule=FillRule.OPEN_OF_CURRENT_BAR,
            expire_after=None,
            participation_cap=None,
            liquidity_lookback_bars=1,
            costs=CostConfig.free(),
        ),
        constraints=ConstraintConfig(
            allow_short=False,
            max_gross_exposure=10.0,
            max_position_weight=10.0,
            cost_buffer_bps=0.0,
        ),
        max_mark_staleness=None,
        end_of_run=EndOfRunPolicy.MARK,
    )
    span = TimeRange(
        start=bars["bar_start"].min(), end=bars["bar_start"].max() + dt.timedelta(minutes=5)
    )
    return run_simulation(
        bars,
        strategy=HoldWhileLastReturnPositive(),
        instruments={SYMBOL: instrument},
        config=config,
        decision_range=span,
        features=compute_features(bars, [LaggedReturn(1)]),
    )


@pytest.fixture(scope="module")
def theirs(bars: pl.DataFrame) -> Any:
    from backtesting import Backtest, Strategy

    frame = (
        bars.select(
            pl.col("bar_start").alias("t"),
            pl.col("open").alias("Open"),
            pl.col("high").alias("High"),
            pl.col("low").alias("Low"),
            pl.col("close").alias("Close"),
            pl.col("volume").alias("Volume"),
        )
        .to_pandas()
        .set_index("t")
    )
    frame.index = frame.index.tz_localize(None)

    class PositiveReturn(Strategy):  # type: ignore[misc]
        def init(self) -> None:
            return None

        def next(self) -> None:
            closes = self.data.Close
            if len(closes) < 2:
                return
            if closes[-1] > closes[-2]:
                if not self.position:
                    self.buy(size=1)
            elif self.position:
                self.position.close()

    stats = Backtest(
        frame,
        PositiveReturn,
        cash=CASH,
        commission=0.0,
        trade_on_close=False,
        exclusive_orders=False,
        finalize_trades=False,
    ).run()
    return frame, stats


def their_fills(theirs: Any) -> Any:
    frame, stats = theirs
    rows = []
    for _, trade in stats["_trades"].iterrows():
        rows.append(
            {
                "at": frame.index[int(trade["EntryBar"])],
                "side": "buy",
                "quantity": abs(float(trade["Size"])),
                "price": float(trade["EntryPrice"]),
            }
        )
        if pd.notna(trade["ExitBar"]):
            rows.append(
                {
                    "at": frame.index[int(trade["ExitBar"])],
                    "side": "sell",
                    "quantity": abs(float(trade["Size"])),
                    "price": float(trade["ExitPrice"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["at", "side"]).reset_index(drop=True)


def our_fills(ours: Any) -> Any:
    return (
        pd.DataFrame(
            [
                {
                    "at": f.fill_at.replace(tzinfo=None) - LATENCY,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                }
                for f in ours.fills
            ]
        )
        .sort_values(["at", "side"])
        .reset_index(drop=True)
    )


# -- the comparison -------------------------------------------------------------------


def test_the_strategy_actually_trades(ours: Any, theirs: Any) -> None:
    """A differential test that compares two empty result sets proves nothing."""
    assert len(ours.fills) > 50, "the fixture must produce a busy strategy"
    assert len(their_fills(theirs)) > 50


def test_both_simulators_produce_the_same_fills(ours: Any, theirs: Any) -> None:
    mine, other = our_fills(ours), their_fills(theirs)
    assert len(mine) == len(other), "fill counts differ"
    assert (mine["side"] == other["side"]).all(), "sides differ"
    assert ((mine["quantity"] - other["quantity"]).abs() < 1e-12).all(), "quantities differ"
    assert ((mine["price"] - other["price"]).abs() < 1e-9).all(), "fill prices differ"


def test_fill_times_differ_only_by_the_publication_latency(ours: Any, theirs: Any) -> None:
    """Documented difference, not disagreement: we stamp at eligibility, they stamp at
    the start of the bar whose open was used."""
    raw = (
        pd.DataFrame([{"at": f.fill_at.replace(tzinfo=None)} for f in ours.fills])
        .sort_values("at")
        .reset_index(drop=True)
    )
    offsets = (raw["at"] - their_fills(theirs)["at"]).unique()
    assert list(offsets) == [pd.Timedelta(LATENCY)]


def test_equity_curves_agree_under_a_common_marking_convention(
    ours: Any, theirs: Any, bars: pl.DataFrame
) -> None:
    """Replay our fills the way backtesting.py marks: the position held *during* bar N,
    valued at close(N)."""
    frame, stats = theirs
    mine = our_fills(ours)
    mine["bar"] = mine["at"].dt.floor("1min")
    by_bar = {bar: group for bar, group in mine.groupby("bar")}

    cash, position, rebuilt = CASH, 0.0, []
    for timestamp, close in zip(frame.index, frame["Close"], strict=True):
        group = by_bar.get(timestamp)
        if group is not None:
            for _, fill in group.iterrows():
                signed = fill["quantity"] * (1 if fill["side"] == "buy" else -1)
                cash -= signed * fill["price"]
                position += signed
        rebuilt.append(cash + position * close)

    reference = stats["_equity_curve"]["Equity"].to_numpy()
    difference = abs(pd.Series(rebuilt) - pd.Series(reference))
    assert difference.max() / CASH < 1e-12, f"max relative difference {difference.max() / CASH:.3g}"


def test_final_equity_matches(ours: Any, theirs: Any) -> None:
    _, stats = theirs
    reference = float(stats["_equity_curve"]["Equity"].iloc[-1])
    assert abs(ours.final.equity - reference) / CASH < 1e-9


def test_the_conservative_rule_is_deliberately_different(
    bars: pl.DataFrame, ours: Any, theirs: Any
) -> None:
    """The bracket is real and quantified: the conservative rule must *not* match the
    reference, and the gap is the cost of the fill assumption."""
    instrument = Instrument(
        instrument_id=SYMBOL,
        asset_class=AssetClass.CRYPTO,
        venue="TEST",
        quote_currency="USDT",
        base_currency="BTC",
        price_increment=Decimal("0.01"),
        quantity_increment=Decimal("0.00001"),
        calendar_id="24x7:1",
    )
    config = SimulationConfig(
        initial_cash=CASH,
        execution=ExecutionConfig(
            submission_latency=dt.timedelta(0),
            order_latency=dt.timedelta(0),
            fill_rule=FillRule.NEXT_OPEN_AFTER_ELIGIBILITY,
            expire_after=None,
            participation_cap=None,
            liquidity_lookback_bars=1,
            costs=CostConfig.free(),
        ),
        constraints=ConstraintConfig(
            allow_short=False,
            max_gross_exposure=10.0,
            max_position_weight=10.0,
            cost_buffer_bps=0.0,
        ),
        max_mark_staleness=None,
        end_of_run=EndOfRunPolicy.MARK,
    )
    span = TimeRange(
        start=bars["bar_start"].min(), end=bars["bar_start"].max() + dt.timedelta(minutes=5)
    )
    conservative = run_simulation(
        bars,
        strategy=HoldWhileLastReturnPositive(),
        instruments={SYMBOL: instrument},
        config=config,
        decision_range=span,
        features=compute_features(bars, [LaggedReturn(1)]),
    )
    _, stats = theirs
    reference = float(stats["_equity_curve"]["Equity"].iloc[-1])
    assert abs(conservative.final.equity - reference) > 1.0, "the two rules must not coincide"
    assert abs(ours.final.equity - reference) / CASH < 1e-9, "but the textbook rule does match"
