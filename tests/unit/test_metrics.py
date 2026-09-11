"""Metrics by hand on a tiny curve, and the annualisation policy."""

from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from qresearch.research.metrics import (
    AnnualizationPolicy,
    annualization_for,
    compute_metrics,
    periodic_equity,
)

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)
TS = pl.Datetime("us", "UTC")


def curve(points: list[tuple[float, float, float]], gross: float = 1.0) -> pl.DataFrame:
    """(minute, second, equity) snapshots with constant exposure fractions."""
    return pl.DataFrame(
        {
            "at": [T0 + dt.timedelta(minutes=m, seconds=s) for m, s, _ in points],
            "equity": [e for _, _, e in points],
            "gross_exposure": [e * gross for _, _, e in points],
            "net_exposure": [e * gross for _, _, e in points],
        },
        schema_overrides={"at": TS},
    )


POLICY = AnnualizationPolicy(periods_per_year=100.0, label="test: 100 periods/year")


def test_periodic_equity_takes_the_last_snapshot_per_bar() -> None:
    out = periodic_equity(
        curve([(0, 0, 100.0), (0, 30, 101.0), (1, 2, 102.0), (1, 40, 103.0)]), "1m"
    )
    assert out.get_column("equity").to_list() == [101.0, 103.0]
    assert out.get_column("period").to_list() == [T0, T0 + dt.timedelta(minutes=1)]


def test_returns_volatility_and_sharpe_by_hand() -> None:
    frames = {"equity_curve": curve([(0, 0, 100.0), (1, 0, 110.0), (2, 0, 99.0), (3, 0, 108.9)])}
    m = compute_metrics(frames, bar_size="1m", annualization=POLICY)
    assert m.periods == 3
    assert m.total_return == pytest.approx(0.089)
    returns = [0.10, -0.10, 0.10]
    mean = sum(returns) / 3
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / 2)
    assert m.annualized_volatility == pytest.approx(std * 10)
    assert m.sharpe == pytest.approx(mean / std * 10)
    assert m.annualized_return == pytest.approx(1.089 ** (100 / 3) - 1)
    assert m.max_drawdown == pytest.approx(0.1)
    assert m.max_drawdown_duration == dt.timedelta(minutes=2), "peak at 00:01, still below at 00:03"


def test_drawdown_duration_measures_peak_to_latest_trough() -> None:
    frames = {
        "equity_curve": curve(
            [(0, 0, 100.0), (1, 0, 90.0), (2, 0, 95.0), (3, 0, 100.0), (4, 0, 120.0)]
        )
    }
    m = compute_metrics(frames, bar_size="1m", annualization=POLICY)
    assert m.max_drawdown == pytest.approx(0.1)
    assert m.max_drawdown_duration == dt.timedelta(minutes=2)


def test_hit_rate_counts_only_exposed_periods() -> None:
    # Exposure 0 during the first period, so the +10% return there does not count.
    frame = curve([(0, 0, 100.0), (1, 0, 110.0), (2, 0, 99.0), (3, 0, 108.9)]).with_columns(
        pl.Series("gross_exposure", [0.0, 110.0, 99.0, 108.9])
    )
    m = compute_metrics({"equity_curve": frame}, bar_size="1m", annualization=POLICY)
    assert m.period_hit_rate == pytest.approx(0.5), "of the two exposed periods, one was up"


def test_costs_and_turnover_from_fills() -> None:
    fills = pl.DataFrame(
        {
            "quantity": [10.0, 10.0],
            "half_spread": [0.02, 0.02],
            "slippage": [0.05, 0.05],
            "fee": [0.1, 0.1],
            "price": [100.0, 110.0],
            "participation": [0.01, None],
        },
    )
    frames = {
        "equity_curve": curve([(0, 0, 1000.0), (1, 0, 1000.0), (2, 0, 1000.0)]),
        "fills": fills,
        "orders": pl.DataFrame({"order_id": ["a", "b"]}),
        "warnings": pl.DataFrame({"occurrences": [2, 3]}),
    }
    m = compute_metrics(frames, bar_size="1m", annualization=POLICY)
    assert m.total_costs == pytest.approx(2 * (10 * 0.07 + 0.1))
    assert m.cost_fraction == pytest.approx(m.total_costs / 1000.0)
    assert m.turnover_annualized == pytest.approx(2100.0 / 1000.0 * 100 / 2)
    assert m.order_count == 2 and m.fill_count == 2
    assert m.mean_participation == pytest.approx(0.01)
    assert m.warning_count == 5


def test_too_few_periods_yield_nulls_not_nonsense() -> None:
    m = compute_metrics(
        {"equity_curve": curve([(0, 0, 100.0)])}, bar_size="1m", annualization=POLICY
    )
    assert m.periods == 0 and m.total_return == 0.0
    assert m.annualized_return is None and m.sharpe is None and m.turnover_annualized is None


def test_flat_returns_have_no_sharpe() -> None:
    m = compute_metrics(
        {"equity_curve": curve([(0, 0, 100.0), (1, 0, 100.0), (2, 0, 100.0)])},
        bar_size="1m",
        annualization=POLICY,
    )
    assert m.sharpe is None and m.annualized_volatility == 0.0


def test_annualisation_policies_are_labelled_and_differ_by_calendar() -> None:
    crypto = annualization_for("24x7:1", "1m")
    equity = annualization_for("XNYS:1", "1m")
    assert crypto.periods_per_year == pytest.approx(365.25 * 1440)
    assert equity.periods_per_year == pytest.approx(252 * 390)
    assert "24x7" in crypto.label and "XNYS" in equity.label
    assert annualization_for("24x7:1", "5m").periods_per_year == pytest.approx(
        crypto.periods_per_year / 5
    )
    with pytest.raises(KeyError, match="no annualisation"):
        annualization_for("XLON:1", "1m")


def test_metrics_are_serialisable() -> None:
    from qresearch.research.metrics import Metrics

    m = compute_metrics(
        {"equity_curve": curve([(0, 0, 100.0), (1, 0, 101.0)])}, bar_size="1m", annualization=POLICY
    )
    assert Metrics.model_validate_json(m.model_dump_json()) == m
