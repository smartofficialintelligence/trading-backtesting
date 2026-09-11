"""Performance metrics with declared annualisation.

Metrics are computed from the persisted ledgers (the frames a simulation writes), not
from the live result object, so a stored run can be re-scored and two runs compared on
identical definitions.

Annualisation is a stated policy (ARCHITECTURE.md sec. 8): crypto and equities cannot
share an unexplained constant. :func:`annualization_for` derives periods-per-year from the
calendar and bar size and labels it; the label is carried on every metrics record.

Definitions (``r_t`` is the per-period equity return, one period per bar interval):

* ``annualized_return``: ``(end / start) ** (periods_per_year / periods) - 1``
* ``annualized_volatility``: ``std(r) * sqrt(periods_per_year)`` (sample std)
* ``sharpe``: ``mean(r - rf_per_period) / std(r) * sqrt(periods_per_year)``
* ``max_drawdown``: largest peak-to-trough equity decline, as a fraction of the peak
* ``turnover_annualized``: ``sum(|fill notional|) / mean(equity) * periods_per_year / periods``
* ``period_hit_rate``: fraction of periods with ``r > 0`` among periods that began with
  non-zero gross exposure. A period-level proxy; round-trip P&L attribution is not
  attempted in the MVP.
* ``cost_fraction``: total friction (spread + slippage + fees) over starting equity
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping

import polars as pl
from pydantic import Field

from qresearch.config import FrozenModel
from qresearch.time import parse_duration

MINUTES_PER_DAY = 1440
_SESSION_MINUTES = {"XNYS": 390}
_TRADING_DAYS = {"XNYS": 252}


class AnnualizationPolicy(FrozenModel):
    periods_per_year: float = Field(gt=0)
    label: str = Field(min_length=1)
    risk_free_rate: float = 0.0
    """Annual, simple. Zero by default and stated as such."""


def annualization_for(calendar_id: str, bar_size: str) -> AnnualizationPolicy:
    """Periods per year for one bar size on one calendar, with a human-readable label."""
    bar_minutes = parse_duration(bar_size) / _dt.timedelta(minutes=1)
    venue = calendar_id.split(":")[0]
    if venue == "24x7":
        minutes_per_year = 365.25 * MINUTES_PER_DAY
        label = f"24x7 calendar, {bar_size} bars: {minutes_per_year / bar_minutes:.0f} periods/year"
    elif venue in _SESSION_MINUTES:
        minutes_per_year = _TRADING_DAYS[venue] * _SESSION_MINUTES[venue]
        label = (
            f"{venue} calendar, {bar_size} bars: {_TRADING_DAYS[venue]} days x "
            f"{_SESSION_MINUTES[venue]} min = {minutes_per_year / bar_minutes:.0f} periods/year"
        )
    else:
        raise KeyError(f"no annualisation policy for calendar {calendar_id!r}")
    return AnnualizationPolicy(periods_per_year=minutes_per_year / bar_minutes, label=label)


class Metrics(FrozenModel):
    periods: int = Field(ge=0)
    start_equity: float
    end_equity: float
    total_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    sharpe: float | None
    max_drawdown: float
    max_drawdown_duration: _dt.timedelta
    turnover_annualized: float | None
    avg_gross_exposure: float
    """As a fraction of equity, averaged over periods."""

    avg_net_exposure: float
    order_count: int
    fill_count: int
    period_hit_rate: float | None
    total_costs: float
    cost_fraction: float
    mean_participation: float | None
    warning_count: int
    annualization: AnnualizationPolicy


def periodic_equity(equity_curve: pl.DataFrame, bar_size: str) -> pl.DataFrame:
    """One row per bar interval: the last snapshot in that interval.

    The engine snapshots at every event instant; metrics need a regular grid. Intervals
    are epoch-aligned like bars, and the interval's value is the state at its end.
    """
    if equity_curve.is_empty():
        return pl.DataFrame(
            schema={
                "period": pl.Datetime("us", "UTC"),
                "equity": pl.Float64,
                "gross_exposure": pl.Float64,
                "net_exposure": pl.Float64,
            }
        )
    step = parse_duration(bar_size)
    return (
        equity_curve.sort("at")
        .with_columns(pl.col("at").dt.truncate(step).alias("period"))
        .group_by("period", maintain_order=True)
        .agg(
            pl.col("equity").last(), pl.col("gross_exposure").last(), pl.col("net_exposure").last()
        )
        .sort("period")
    )


def compute_metrics(
    frames: Mapping[str, pl.DataFrame],
    *,
    bar_size: str,
    annualization: AnnualizationPolicy,
) -> Metrics:
    """Score one simulation from its ledgers (see ``SimulationResult.frames``)."""
    curve = periodic_equity(frames["equity_curve"], bar_size)
    fills = frames.get("fills", pl.DataFrame())
    orders = frames.get("orders", pl.DataFrame())
    warnings = frames.get("warnings", pl.DataFrame())
    ppy = annualization.periods_per_year

    equity = (
        curve.get_column("equity") if not curve.is_empty() else pl.Series("equity", [], pl.Float64)
    )
    n = max(equity.len() - 1, 0)
    start = float(equity[0]) if equity.len() else 0.0
    end = float(equity[-1]) if equity.len() else 0.0
    total_return = end / start - 1.0 if start > 0 else 0.0

    returns = (
        (equity / equity.shift(1) - 1.0).drop_nulls()
        if equity.len() > 1
        else pl.Series([], dtype=pl.Float64)
    )
    rf_period = annualization.risk_free_rate / ppy
    ann_return = (end / start) ** (ppy / n) - 1.0 if n > 0 and start > 0 else None
    std = float(returns.std()) if returns.len() >= 2 else None  # type: ignore[arg-type]
    ann_vol = std * math.sqrt(ppy) if std is not None else None
    sharpe = (
        float((returns - rf_period).mean()) / std * math.sqrt(ppy)  # type: ignore[arg-type]
        if std is not None and std > 0
        else None
    )

    max_dd, dd_duration = _drawdown(curve)

    total_costs = 0.0
    fill_notional = 0.0
    mean_participation = None
    if not fills.is_empty():
        total_costs = float(
            (fills["quantity"] * (fills["half_spread"] + fills["slippage"]) + fills["fee"]).sum()
        )
        fill_notional = float((fills["quantity"] * fills["price"]).sum())
        participation = fills["participation"].drop_nulls()
        mean_participation = float(participation.mean()) if participation.len() else None  # type: ignore[arg-type]

    mean_equity = float(equity.mean()) if equity.len() else 0.0  # type: ignore[arg-type]
    turnover = fill_notional / mean_equity * ppy / n if n > 0 and mean_equity > 0 else None

    gross = (
        curve.get_column("gross_exposure")
        if not curve.is_empty()
        else pl.Series([], dtype=pl.Float64)
    )
    net = (
        curve.get_column("net_exposure")
        if not curve.is_empty()
        else pl.Series([], dtype=pl.Float64)
    )
    avg_gross = float((gross / equity).mean()) if equity.len() else 0.0  # type: ignore[arg-type]
    avg_net = float((net / equity).mean()) if equity.len() else 0.0  # type: ignore[arg-type]

    hit_rate = None
    if n > 0:
        exposed = (gross.shift(1) > 0).drop_nulls()
        wins = (returns > 0).filter(exposed.to_list())
        hit_rate = float(wins.mean()) if wins.len() else None  # type: ignore[arg-type]

    return Metrics(
        periods=n,
        start_equity=start,
        end_equity=end,
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        turnover_annualized=turnover,
        avg_gross_exposure=avg_gross,
        avg_net_exposure=avg_net,
        order_count=orders.height,
        fill_count=fills.height,
        period_hit_rate=hit_rate,
        total_costs=total_costs,
        cost_fraction=total_costs / start if start > 0 else 0.0,
        mean_participation=mean_participation,
        warning_count=int(warnings["occurrences"].sum()) if not warnings.is_empty() else 0,
        annualization=annualization,
    )


def _drawdown(curve: pl.DataFrame) -> tuple[float, _dt.timedelta]:
    if curve.is_empty():
        return 0.0, _dt.timedelta(0)
    equity = curve.get_column("equity").to_list()
    periods = curve.get_column("period").to_list()
    peak = equity[0]
    peak_at = periods[0]
    max_dd = 0.0
    longest = _dt.timedelta(0)
    for value, at in zip(equity, periods, strict=True):
        if value >= peak:
            peak, peak_at = value, at
        else:
            max_dd = max(max_dd, 1.0 - value / peak)
            longest = max(longest, at - peak_at)
    return max_dd, longest
