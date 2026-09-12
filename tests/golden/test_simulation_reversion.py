"""Z-score reversion through the engine: one entry, no churn while holding, one exit."""

from __future__ import annotations

import polars as pl
from tests.golden.conftest import at, bars, free_config, instrument

from qresearch.features.pipeline import compute_features
from qresearch.features.technical import RollingZScore
from qresearch.research.splits import TimeRange
from qresearch.simulation.engine import run_simulation
from qresearch.strategy.reversion import ZScoreReversion


def _crash_path() -> list[tuple[int, float, float, float]]:
    """Twenty bars wiggling around 100, then a gap down to 90 where it stays.

    The rolling mean catches down to the new level, so z goes from about -2.8 on the
    crash bar back through zero around bar 29 -- while price keeps moving by ±0.5 each
    bar, which is what would make a target-weight strategy trade every bar.
    """
    return [
        (m, level, level + (0.5 if m % 2 else -0.5), 1_000.0)
        for m in range(34)
        for level in [100.0 if m < 20 else 90.0]
    ]


def test_a_crash_is_bought_once_held_without_trading_and_sold_back_at_the_mean() -> None:
    frame = bars("X", _crash_path())
    features = compute_features(frame, [RollingZScore(window=10)])
    result = run_simulation(
        frame,
        strategy=ZScoreReversion(zscore="zscore_10", entry_z=2.5, sides="long"),
        instruments={"X": instrument("X")},
        config=free_config(),
        decision_range=TimeRange(start=at(0), end=at(100)),
        features=features,
    )
    z = features.select("available_at", "zscore_10").drop_nulls().sort("available_at")
    entry_signal = z.filter(pl.col("zscore_10") <= -2.5)["available_at"][0]

    assert [o.side.value for o in result.orders] == ["buy", "sell"], "no top-ups or trims"
    assert result.orders[0].signal_at == entry_signal
    entry_fill = result.fills[0].fill_at
    exit_signal = z.filter((pl.col("available_at") > entry_fill) & (pl.col("zscore_10") >= 0))[
        "available_at"
    ][0]
    assert result.orders[1].signal_at == exit_signal
    assert result.final.position_count == 0
