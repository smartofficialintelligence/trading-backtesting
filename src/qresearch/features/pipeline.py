"""Compute a set of features over a bar frame, with availability owned centrally.

The pipeline is where the timing contract in ARCHITECTURE.md sec. 2 is applied to
features: "a feature derived from several observations is available no earlier than the
maximum ``available_at`` of all its inputs plus any declared computation latency". Each
feature declares its lookback; the pipeline turns that into a rolling maximum over the
input bars' ``available_at``, so no feature author ever computes availability by hand.

One ``available_at`` per output row, equal to the latest across all features in the set
(and never earlier than the bar's own). A strategy receives the row as one batch
(ARCHITECTURE.md sec. 2, "batch simultaneous information"); a feature whose window is not
yet complete shows as null, which is the honest answer at that instant.

Gaps: a feature's window is positional (the previous ``lookback`` rows for that
instrument). When those rows are not consecutive in time -- a missing bar, a halt, a
session break -- the value is nulled unless the spec opts out. A "5-bar return" that
silently spans an overnight gap is a different quantity from the one the name claims.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence

import polars as pl

from qresearch.features.contracts import FEATURE_KEY, Feature
from qresearch.time import parse_duration

INSTRUMENT = FEATURE_KEY[0]


def compute_features(
    bars: pl.DataFrame | pl.LazyFrame,
    features: Sequence[Feature],
) -> pl.DataFrame:
    """Evaluate ``features`` over canonical bars.

    Args:
        bars: a canonical bar frame (one ``bar_size``) as produced by the catalog. Row
            order does not matter; the result is sorted by ``(instrument_id, bar_start)``.
        features: the feature set. Names must be unique.

    Returns:
        One row per input bar with columns ``instrument_id``, ``bar_start``, one column
        per feature, and ``available_at`` -- the instant the whole row may be consumed.
    """
    frame = bars.collect() if isinstance(bars, pl.LazyFrame) else bars
    if frame.is_empty():
        raise ValueError("cannot compute features over an empty bar frame")
    if not features:
        raise ValueError("no features requested")

    names = [f.spec.name for f in features]
    if len(set(names)) != len(names):
        raise ValueError(
            f"duplicate feature names: {sorted(n for n in names if names.count(n) > 1)}"
        )
    for column in ("available_at", "bar_size", *FEATURE_KEY):
        if column not in frame.columns:
            raise ValueError(f"bar frame lacks required column {column!r}")
    required = {c for f in features for c in f.spec.inputs}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise ValueError(f"features require bar columns {absent} which are not present")

    sizes = frame.get_column("bar_size").unique().to_list()
    if len(sizes) != 1:
        raise ValueError(f"bar frame must have exactly one bar_size, found {sorted(sizes)}")
    step = parse_duration(sizes[0])

    ordered = frame.sort([*FEATURE_KEY])
    value_exprs: list[pl.Expr] = []
    availability_exprs: list[pl.Expr] = [pl.col("available_at")]
    for feature in features:
        spec = feature.spec
        value = feature.expression()
        if spec.require_contiguous and spec.lookback_bars > 0:
            value = pl.when(_contiguous(spec.lookback_bars, step)).then(value).otherwise(None)
        value_exprs.append(value.over(INSTRUMENT).alias(spec.name))
        availability_exprs.append(
            _window_availability(spec.window_bars, spec.computation_latency)
            .over(INSTRUMENT)
            .alias(f"__avail_{spec.name}")
        )

    return (
        ordered.with_columns(*value_exprs, *availability_exprs)
        .with_columns(
            pl.max_horizontal(
                pl.col("available_at"), *[pl.col(f"__avail_{n}") for n in names]
            ).alias("__row_available_at")
        )
        .select(*FEATURE_KEY, *names, pl.col("__row_available_at").alias("available_at"))
    )


def _contiguous(lookback: int, step: _dt.timedelta) -> pl.Expr:
    """True when the previous ``lookback`` bars are consecutive on the grid."""
    micros = (step * lookback) // _dt.timedelta(microseconds=1)
    return (pl.col("bar_start") - pl.col("bar_start").shift(lookback)) == pl.duration(
        microseconds=micros
    )


def _window_availability(window_bars: int, latency: _dt.timedelta) -> pl.Expr:
    """Latest input availability across the window, plus computation latency.

    ``min_samples=window_bars`` leaves this null during warm-up, where the value is null
    too; the row then falls back to the bar's own availability via ``max_horizontal``.
    """
    micros = latency // _dt.timedelta(microseconds=1)
    return pl.col("available_at").rolling_max(
        window_size=window_bars, min_samples=window_bars
    ) + pl.duration(microseconds=micros)
