"""1-minute to 5-minute derivation.

DEVELOPMENT_PLAN.md Stage 1 acceptance: "Five-minute bars never appear before every
included minute is available." This is the resampling-leakage case from ARCHITECTURE.md
sec. 9 -- a partial window treated as complete, or a window stamped with its nominal
boundary rather than the availability of its slowest input.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from qresearch.application.ingest import resample_bars

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)


def minute_bars(
    count: int, *, late: dict[int, dt.timedelta] | None = None, skip: set[int] | None = None
) -> pl.DataFrame:
    """Build ``count`` one-minute bars, optionally delaying or omitting some."""
    late = late or {}
    skip = skip or set()
    rows = []
    for i in range(count):
        if i in skip:
            continue
        start = T0 + dt.timedelta(minutes=i)
        end = start + dt.timedelta(minutes=1)
        rows.append(
            {
                "instrument_id": "X",
                "bar_size": "1m",
                "bar_start": start,
                "bar_end": end,
                "available_at": end + late.get(i, dt.timedelta(seconds=2)),
                "open": 100.0 + i,
                "high": 100.5 + i,
                "low": 99.5 + i,
                "close": 100.2 + i,
                "volume": 10.0 + i,
                "vwap": 100.0 + i,
                "trade_count": 5,
                "source": "test",
                "source_key": None,
                "revision": 0,
                "ingested_at": T0,
            }
        )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "bar_start": pl.Datetime("us", "UTC"),
            "bar_end": pl.Datetime("us", "UTC"),
            "available_at": pl.Datetime("us", "UTC"),
            "ingested_at": pl.Datetime("us", "UTC"),
            "revision": pl.Int32,
            "trade_count": pl.Int64,
        },
    )


def test_ohlcv_aggregates_correctly() -> None:
    out = resample_bars(minute_bars(5), target_bar_size="5m").row(0, named=True)
    assert out["open"] == 100.0
    assert out["close"] == pytest.approx(104.2)
    assert out["high"] == pytest.approx(104.5)
    assert out["low"] == pytest.approx(99.5)
    assert out["volume"] == pytest.approx(sum(10.0 + i for i in range(5)))


def test_the_window_is_half_open_and_epoch_aligned() -> None:
    out = resample_bars(minute_bars(10), target_bar_size="5m").sort("bar_start")
    assert out.height == 2
    assert out.get_column("bar_start").to_list() == [T0, T0 + dt.timedelta(minutes=5)]
    assert out.item(0, "bar_end") == T0 + dt.timedelta(minutes=5)
    assert out.item(0, "bar_size") == "5m"


def test_availability_is_the_slowest_input_not_the_window_boundary() -> None:
    """One late minute delays the whole 5-minute bar.

    Stamping the window's nominal end here would publish a bar summarising data that had
    not arrived -- a one-sided leak that looks like nothing at all in the output.
    """
    out = resample_bars(
        minute_bars(5, late={2: dt.timedelta(minutes=9)}), target_bar_size="5m"
    ).row(0, named=True)
    expected = T0 + dt.timedelta(minutes=3) + dt.timedelta(minutes=9)
    assert out["available_at"] == expected
    assert out["available_at"] > out["bar_end"]


def test_availability_never_precedes_the_window_end_when_inputs_are_complete() -> None:
    out = resample_bars(minute_bars(20), target_bar_size="5m")
    assert (out.get_column("available_at") >= out.get_column("bar_end")).all()


def test_an_incomplete_window_publishes_no_earlier_than_its_window_end() -> None:
    """A window missing its final minutes must not be offered before the window closes.

    Inputs run to minute 2, so the slowest input is available at 00:03:02 -- but minutes 3
    and 4 could still arrive late and belong to this same window. Publishing at 00:03:02
    would show a consumer a coarse bar that is not final yet.
    """
    out = resample_bars(minute_bars(5, skip={3, 4}), target_bar_size="5m").row(0, named=True)
    assert out["available_at"] == T0 + dt.timedelta(minutes=5)
    assert out["available_at"] == out["bar_end"]
    assert out["volume"] == pytest.approx(10.0 + 11.0 + 12.0), "gaps are not filled"


def test_a_late_input_still_dominates_the_window_end() -> None:
    """The clamp is a floor, not a replacement: a slow input still delays the bar."""
    out = resample_bars(
        minute_bars(5, late={2: dt.timedelta(minutes=9)}), target_bar_size="5m"
    ).row(0, named=True)
    assert out["available_at"] > out["bar_end"]


def test_vwap_is_volume_weighted() -> None:
    frame = minute_bars(5)
    out = resample_bars(frame, target_bar_size="5m").row(0, named=True)
    expected = (frame.get_column("vwap") * frame.get_column("volume")).sum() / frame.get_column(
        "volume"
    ).sum()
    assert out["vwap"] == pytest.approx(expected)


def test_instruments_are_resampled_independently() -> None:
    a = minute_bars(5)
    b = minute_bars(5).with_columns(pl.lit("Y").alias("instrument_id"))
    out = resample_bars(pl.concat([a, b]), target_bar_size="5m")
    assert out.height == 2
    assert sorted(out.get_column("instrument_id").to_list()) == ["X", "Y"]


@pytest.mark.parametrize("target", ["1m", "30s", "90s"])
def test_a_smaller_or_non_multiple_target_is_refused(target: str) -> None:
    """Upsampling would invent data; a non-multiple would split a source bar in two."""
    with pytest.raises(ValueError, match="whole multiple"):
        resample_bars(minute_bars(5), target_bar_size=target)


def test_windows_are_aligned_to_the_epoch_not_to_the_first_observation() -> None:
    """Alignment is absolute, so two datasets over the same period bucket identically.

    5m divides the 1440 minutes of a UTC day, so a 5m window always starts at midnight.
    """
    out = resample_bars(minute_bars(10), target_bar_size="5m").sort("bar_start")
    assert out.get_column("bar_start").to_list() == [T0, T0 + dt.timedelta(minutes=5)]

    # 7m does not divide a day, so midnight falls mid-window and the first bucket is
    # partial. That is the honest result: the window is where the epoch grid puts it, not
    # where the data happens to start.
    seven = resample_bars(minute_bars(14), target_bar_size="7m").sort("bar_start")
    assert seven.item(0, "bar_start") < T0
    assert (seven.get_column("bar_end") - seven.get_column("bar_start")).unique().to_list() == [
        dt.timedelta(minutes=7)
    ]
    assert (seven.get_column("available_at") >= seven.get_column("bar_end")).all()


def test_mixed_source_bar_sizes_are_refused() -> None:
    mixed = pl.concat([minute_bars(5), minute_bars(5).with_columns(pl.lit("5m").alias("bar_size"))])
    with pytest.raises(ValueError, match="exactly one source bar_size"):
        resample_bars(mixed, target_bar_size="5m")
