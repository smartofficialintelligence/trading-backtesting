"""Executable leakage checks for feature computations.

DEVELOPMENT_PLAN.md Stage 2 asks for "prefix-invariance and sentinel-future test helpers
reusable by all features". They live in the package rather than under ``tests/`` because
a researcher writing a feature in a notebook should be able to run them too.

Both take a ``compute`` callable -- typically
``lambda bars: compute_features(bars, [feature])`` -- and a canonical bar frame, and raise
:class:`LeakageDetected` with the offending rows when the invariant fails.

Float comparison uses a relative tolerance (``rtol``, default ``1e-9``). Polars' rolling
statistics are streaming algorithms whose last-bit rounding depends on the values
processed *before* the window, so a mathematically causal ``rolling_std`` is not
bit-identical between a prefix and the full series. The checks are looking for
information leaks, and a leak never shows up as a last-bit effect -- a future price read
into a window moves the result by orders of magnitude more. Pass ``rtol=0`` to demand
exact reproduction.

The two checks are complementary:

* :func:`assert_prefix_invariant` removes the future and recomputes. It catches features
  whose *availability* is wrong (claims to be usable before its inputs were) and features
  that depend on rows that were not there yet (centred windows, negative shifts, global
  aggregates).
* :func:`assert_future_insensitive` keeps the future rows but corrupts their values. It
  catches features that report availability correctly but still *read* later rows -- the
  case a prefix check can miss when a feature nulls itself gracefully at the data edge.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Sequence

import polars as pl

from qresearch.features.contracts import FEATURE_KEY

Compute = Callable[[pl.DataFrame], pl.DataFrame]

SENTINEL_PRICE = 1.0e9
SENTINEL_VOLUME = 1.0e12
DEFAULT_RTOL = 1e-9
_VALUE_COLUMNS = ("open", "high", "low", "close", "volume", "vwap", "trade_count")


class LeakageDetected(AssertionError):
    """A feature's output at or before a cutoff depended on data after it."""

    def __init__(self, message: str, offending: pl.DataFrame) -> None:
        super().__init__(f"{message}\n{offending.head(10)}")
        self.offending = offending


def default_cut_points(bars: pl.DataFrame, count: int = 5) -> tuple[_dt.datetime, ...]:
    """Evenly spaced cutoffs across the frame's availability range, excluding the ends."""
    series = bars.get_column("available_at").sort()
    n = series.len()
    if n < 3:
        raise ValueError("need at least three bars to choose interior cut points")
    picks = sorted({int(n * (i + 1) / (count + 1)) for i in range(count)})
    return tuple(series[i] for i in picks)


def assert_prefix_invariant(
    compute: Compute,
    bars: pl.DataFrame,
    *,
    cut_points: Sequence[_dt.datetime] | None = None,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """Recomputing on only the bars available by ``t`` must reproduce every output
    row whose ``available_at <= t``.

    The comparison set is defined by the *full* computation's availability. That is the
    contract being tested: a feature's declared availability is a promise that all its
    inputs existed by then, so the prefix must be able to reproduce it exactly.
    """
    full = compute(bars)
    _require_shape(full)
    cuts = tuple(cut_points) if cut_points is not None else default_cut_points(bars)
    value_columns = [c for c in full.columns if c not in (*FEATURE_KEY, "available_at")]

    for cut in cuts:
        expected = full.filter(pl.col("available_at") <= cut)
        if expected.is_empty():
            continue
        prefix = compute(bars.filter(pl.col("available_at") <= cut))
        _require_shape(prefix)
        joined = expected.join(prefix, on=list(FEATURE_KEY), how="left", suffix="__prefix")

        absent = joined.filter(pl.col("available_at__prefix").is_null())
        if not absent.is_empty():
            raise LeakageDetected(
                f"at cut {cut}, {absent.height} row(s) the full computation marks "
                "available are missing from the prefix computation; the feature claims "
                "availability before its inputs existed",
                absent.select([*FEATURE_KEY, "available_at"]),
            )
        for column in value_columns:
            differing = joined.filter(_differs(joined, column, f"{column}__prefix", rtol=rtol))
            if not differing.is_empty():
                raise LeakageDetected(
                    f"at cut {cut}, feature {column!r} differs between the full and "
                    f"prefix computations on {differing.height} row(s) marked available "
                    "by then; the value depends on data after its declared availability",
                    differing.select([*FEATURE_KEY, "available_at", column, f"{column}__prefix"]),
                )


def assert_future_insensitive(
    compute: Compute,
    bars: pl.DataFrame,
    *,
    cut_points: Sequence[_dt.datetime] | None = None,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """Corrupting every value in bars available after ``t`` must not change any
    output row whose ``available_at <= t``.

    Rows are kept (so positional windows are unchanged) but their prices and volumes are
    replaced with absurd sentinels. Any output at or before the cut that moves was reading
    the future.
    """
    full = compute(bars)
    _require_shape(full)
    cuts = tuple(cut_points) if cut_points is not None else default_cut_points(bars)
    value_columns = [c for c in full.columns if c not in (*FEATURE_KEY, "available_at")]
    present = [c for c in _VALUE_COLUMNS if c in bars.columns]

    for cut in cuts:
        is_future = pl.col("available_at") > cut
        corrupted = bars.with_columns(
            *[
                pl.when(is_future)
                .then(pl.lit(SENTINEL_VOLUME if c in ("volume", "trade_count") else SENTINEL_PRICE))
                .otherwise(pl.col(c))
                .cast(bars.schema[c])
                .alias(c)
                for c in present
            ]
        )
        expected = full.filter(pl.col("available_at") <= cut)
        if expected.is_empty():
            continue
        actual = compute(corrupted)
        _require_shape(actual)
        joined = expected.join(actual, on=list(FEATURE_KEY), how="left", suffix="__corrupt")
        for column in value_columns:
            differing = joined.filter(_differs(joined, column, f"{column}__corrupt", rtol=rtol))
            if not differing.is_empty():
                raise LeakageDetected(
                    f"at cut {cut}, feature {column!r} changed on {differing.height} row(s) "
                    "marked available by then when only later bars were altered; the value "
                    "reads data after its declared availability",
                    differing.select([*FEATURE_KEY, "available_at", column, f"{column}__corrupt"]),
                )
        moved = joined.filter(pl.col("available_at__corrupt") != pl.col("available_at"))
        if not moved.is_empty():
            raise LeakageDetected(
                f"at cut {cut}, availability of {moved.height} row(s) changed when only "
                "later bars were altered",
                moved.select([*FEATURE_KEY, "available_at", "available_at__corrupt"]),
            )


def _differs(frame: pl.DataFrame, a: str, b: str, *, rtol: float) -> pl.Expr:
    """Null-aware inequality: null == null and NaN == NaN count as equal; null != value
    counts as different; floats within ``rtol`` (relative) count as equal.

    Must never itself evaluate to null: ``filter`` treats null as False, and a
    null-propagating ``==`` would silently drop the very rows -- value on one side, null
    on the other -- that a leaking feature produces at the edge of the prefix.
    """
    if rtol < 0:
        raise ValueError(f"rtol must be >= 0, got {rtol}")
    equal = pl.col(a).eq_missing(pl.col(b))
    if frame.schema[a].is_float():
        both_nan = (pl.col(a).is_nan() & pl.col(b).is_nan()).fill_null(False)
        scale = pl.max_horizontal(pl.col(a).abs(), pl.col(b).abs())
        within = ((pl.col(a) - pl.col(b)).abs() <= rtol * scale).fill_null(False)
        equal = equal | both_nan | within
    return ~equal


def _require_shape(frame: pl.DataFrame) -> None:
    for column in (*FEATURE_KEY, "available_at"):
        if column not in frame.columns:
            raise ValueError(
                f"compute() must return a frame with {FEATURE_KEY} and 'available_at'; "
                f"got columns {frame.columns}"
            )
