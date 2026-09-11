"""Flat-to-flat trade attribution, checked against hand-computed P&L and the portfolio."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from qresearch.research.trades import build_trades, trade_stats

T0 = dt.datetime(2024, 3, 4, tzinfo=dt.UTC)
TS = pl.Datetime("us", "UTC")


def fills(*rows: tuple[int, str, str, float, float, float, float]) -> pl.DataFrame:
    """(minute, instrument, side, quantity, reference_price, friction_per_unit, fee)."""
    return pl.DataFrame(
        {
            "fill_id": [f"f{i}" for i in range(len(rows))],
            "instrument_id": [r[1] for r in rows],
            "side": [r[2] for r in rows],
            "quantity": [r[3] for r in rows],
            "fill_at": [T0 + dt.timedelta(minutes=r[0]) for r in rows],
            "reference_price": [r[4] for r in rows],
            "half_spread": [r[5] for r in rows],
            "slippage": [0.0 for _ in rows],
            # price = reference + side * (half_spread + slippage)
            "price": [r[4] + (1 if r[2] == "buy" else -1) * r[5] for r in rows],
            "fee": [r[6] for r in rows],
        },
        schema_overrides={"fill_at": TS},
    )


# -- grouping --------------------------------------------------------------------------


def test_a_simple_round_trip_is_one_trade() -> None:
    t = build_trades(
        fills((0, "X", "buy", 10, 100.0, 0.0, 0.0), (5, "X", "sell", 10, 110.0, 0.0, 0.0))
    )
    assert t.height == 1
    row = t.row(0, named=True)
    assert row["direction"] == "long" and not row["is_open"]
    assert row["opened_at"] == T0 and row["closed_at"] == T0 + dt.timedelta(minutes=5)
    assert row["duration_s"] == 300.0
    assert row["peak_quantity"] == 10.0 and row["fill_count"] == 2
    assert row["entry_price"] == 100.0 and row["exit_price"] == 110.0
    assert row["net_pnl"] == pytest.approx(100.0)
    assert row["return_on_notional"] == pytest.approx(100.0 / 1000.0)


def test_adds_and_trims_belong_to_one_trade() -> None:
    """The consequence of flat-to-flat: scaling in and out is a single episode."""
    t = build_trades(
        fills(
            (0, "X", "buy", 10, 100.0, 0.0, 0.0),
            (1, "X", "buy", 10, 102.0, 0.0, 0.0),
            (2, "X", "sell", 5, 108.0, 0.0, 0.0),
            (3, "X", "sell", 15, 110.0, 0.0, 0.0),
        )
    )
    assert t.height == 1
    row = t.row(0, named=True)
    assert row["fill_count"] == 4 and row["peak_quantity"] == 20.0
    # -(10*100 + 10*102 - 5*108 - 15*110) = -(1000 + 1020 - 540 - 1650) = 170
    assert row["net_pnl"] == pytest.approx(170.0)
    assert row["entry_price"] == pytest.approx(101.0), "volume-weighted entry"


def test_a_flip_closes_one_trade_and_opens_another() -> None:
    t = build_trades(
        fills(
            (0, "X", "buy", 10, 100.0, 0.0, 0.0),
            (5, "X", "sell", 15, 110.0, 0.0, 0.0),  # closes long 10, opens short 5
            (9, "X", "buy", 5, 105.0, 0.0, 0.0),
        )
    ).sort("trade_id")
    assert t.height == 2
    long_trade, short_trade = t.row(0, named=True), t.row(1, named=True)
    assert long_trade["direction"] == "long" and long_trade["net_pnl"] == pytest.approx(100.0)
    assert short_trade["direction"] == "short"
    assert short_trade["opened_at"] == T0 + dt.timedelta(minutes=5)
    assert short_trade["peak_quantity"] == 5.0
    assert short_trade["net_pnl"] == pytest.approx(5 * (110.0 - 105.0))


def test_instruments_are_independent() -> None:
    t = build_trades(
        fills(
            (0, "X", "buy", 10, 100.0, 0.0, 0.0),
            (0, "Y", "buy", 5, 50.0, 0.0, 0.0),
            (5, "X", "sell", 10, 110.0, 0.0, 0.0),
            (6, "Y", "sell", 5, 45.0, 0.0, 0.0),
        )
    )
    assert t.height == 2
    by_instrument = {r["instrument_id"]: r for r in t.iter_rows(named=True)}
    assert by_instrument["X"]["net_pnl"] == pytest.approx(100.0)
    assert by_instrument["Y"]["net_pnl"] == pytest.approx(-25.0)


def test_a_short_trade() -> None:
    t = build_trades(
        fills((0, "X", "sell", 10, 100.0, 0.0, 0.0), (5, "X", "buy", 10, 90.0, 0.0, 0.0))
    )
    row = t.row(0, named=True)
    assert row["direction"] == "short"
    assert row["net_pnl"] == pytest.approx(100.0), "shorted high, covered low"
    assert row["entry_price"] == 100.0 and row["exit_price"] == 90.0


# -- costs -----------------------------------------------------------------------------


def test_the_pnl_identity_holds() -> None:
    """net = gross - costs, where gross is measured against the untouched reference."""
    t = build_trades(
        fills(
            (0, "X", "buy", 10, 100.0, 0.05, 1.0),
            (5, "X", "sell", 10, 110.0, 0.05, 1.0),
        )
    )
    row = t.row(0, named=True)
    assert row["gross_pnl"] == pytest.approx(100.0), "versus the reference price"
    # 10 * 0.05 spread on each side, plus 1.0 fee each side
    assert row["costs"] == pytest.approx(0.5 + 1.0 + 0.5 + 1.0)
    assert row["net_pnl"] == pytest.approx(row["gross_pnl"] - row["costs"])
    assert row["net_pnl"] == pytest.approx(97.0)


def test_costs_are_split_across_trades_when_a_fill_spans_a_flip() -> None:
    """A flipping fill pays for both sides; each trade takes its share."""
    t = build_trades(
        fills(
            (0, "X", "buy", 10, 100.0, 0.0, 0.0),
            (5, "X", "sell", 20, 110.0, 0.0, 6.0),  # 10 closes, 10 opens -> fee splits evenly
            (9, "X", "buy", 10, 110.0, 0.0, 0.0),
        )
    ).sort("trade_id")
    assert t.get_column("costs").to_list() == pytest.approx([3.0, 3.0])


# -- open trades -------------------------------------------------------------------------


def test_an_open_trade_is_marked_and_flagged() -> None:
    t = build_trades(fills((0, "X", "buy", 10, 100.0, 0.0, 0.0)), marks={"X": 107.0})
    row = t.row(0, named=True)
    assert row["is_open"] and row["closed_at"] is None
    assert row["net_pnl"] == pytest.approx(70.0), "marked at 107"
    assert row["exit_price"] == pytest.approx(107.0)


def test_an_open_trade_without_a_mark_reports_unknown_pnl_not_the_cash_outflow() -> None:
    """The cash outflow alone (-1000 here) reads as a total loss; zero would be a guess.
    An unvalued position has unknown P&L and must say so."""
    t = build_trades(fills((0, "X", "buy", 10, 100.0, 0.0, 0.0)))
    row = t.row(0, named=True)
    assert row["is_open"]
    assert row["net_pnl"] is None and row["gross_pnl"] is None
    assert row["return_on_notional"] is None and row["exit_price"] is None
    assert row["costs"] == 0.0, "costs are known even when the P&L is not"
    assert row["entry_price"] == 100.0 and row["peak_quantity"] == 10.0


def test_open_trades_are_excluded_from_statistics() -> None:
    t = build_trades(
        fills(
            (0, "X", "buy", 10, 100.0, 0.0, 0.0),
            (5, "X", "sell", 10, 110.0, 0.0, 0.0),
            (6, "Y", "buy", 10, 100.0, 0.0, 0.0),
        ),
        marks={"Y": 200.0},
    )
    stats = trade_stats(t)
    assert stats.trade_count == 1, "only the closed trade"
    assert stats.open_at_end == 1
    assert stats.net_pnl == pytest.approx(100.0), "the open trade's +1000 is excluded"


# -- statistics --------------------------------------------------------------------------


def test_statistics_by_hand() -> None:
    t = build_trades(
        fills(
            (0, "X", "buy", 1, 100.0, 0.0, 0.0),
            (1, "X", "sell", 1, 110.0, 0.0, 0.0),  # +10
            (2, "X", "buy", 1, 100.0, 0.0, 0.0),
            (3, "X", "sell", 1, 130.0, 0.0, 0.0),  # +30
            (4, "X", "buy", 1, 100.0, 0.0, 0.0),
            (5, "X", "sell", 1, 92.0, 0.0, 0.0),  # -8
        )
    )
    stats = trade_stats(t)
    assert stats.trade_count == 3
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.average_win == pytest.approx(20.0)
    assert stats.average_loss == pytest.approx(-8.0)
    assert stats.profit_factor == pytest.approx(40.0 / 8.0)
    assert stats.expectancy == pytest.approx(32.0 / 3)
    assert stats.largest_win == pytest.approx(30.0) and stats.largest_loss == pytest.approx(-8.0)
    assert stats.long_count == 3 and stats.short_count == 0
    assert stats.median_duration_s == 60.0


def test_profit_factor_is_none_rather_than_infinite_when_nothing_lost() -> None:
    t = build_trades(
        fills((0, "X", "buy", 1, 100.0, 0.0, 0.0), (1, "X", "sell", 1, 110.0, 0.0, 0.0))
    )
    stats = trade_stats(t)
    assert stats.profit_factor is None, "undefined, not inf -- inf as a headline is worse"
    assert stats.win_rate == 1.0


def test_empty_inputs() -> None:
    empty = build_trades(
        pl.DataFrame(
            schema={
                "fill_id": pl.String,
                "instrument_id": pl.String,
                "side": pl.String,
                "quantity": pl.Float64,
                "fill_at": TS,
                "reference_price": pl.Float64,
                "price": pl.Float64,
                "fee": pl.Float64,
            }
        )
    )
    assert empty.is_empty()
    stats = trade_stats(empty)
    assert stats.trade_count == 0 and stats.win_rate is None


def test_missing_columns_are_refused() -> None:
    with pytest.raises(ValueError, match="lacks required column"):
        build_trades(pl.DataFrame({"instrument_id": ["X"]}))


def test_stats_are_serialisable() -> None:
    t = build_trades(
        fills((0, "X", "buy", 1, 100.0, 0.0, 0.0), (1, "X", "sell", 1, 110.0, 0.0, 0.0))
    )
    stats = trade_stats(t)
    assert type(stats).model_validate_json(stats.model_dump_json()) == stats
