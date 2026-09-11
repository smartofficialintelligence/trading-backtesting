"""The sanctioned as-of join: matches on availability, never on event time."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from qresearch.data.point_in_time import AvailabilityViolation, asof_join

T0 = dt.datetime(2024, 3, 4, 0, 0, tzinfo=dt.UTC)
TS = pl.Datetime("us", "UTC")


def at(minute: int, second: int = 0) -> dt.datetime:
    return T0 + dt.timedelta(minutes=minute, seconds=second)


def decisions(*rows: tuple[str, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"instrument_id": [r[0] for r in rows], "as_of": [at(r[1]) for r in rows]},
        schema_overrides={"as_of": TS},
    )


def observations(*rows: tuple[str, int, float]) -> pl.DataFrame:
    """(instrument, available minute, value)."""
    return pl.DataFrame(
        {
            "instrument_id": [r[0] for r in rows],
            "available_at": [at(r[1]) for r in rows],
            "rate": [r[2] for r in rows],
        },
        schema_overrides={"available_at": TS},
    )


def test_attaches_the_latest_observation_available_by_the_cutoff() -> None:
    out = asof_join(
        decisions(("a", 10)),
        observations(("a", 4, 1.0), ("a", 8, 2.0), ("a", 12, 3.0)),
        as_of_col="as_of",
    )
    assert out.item(0, "rate") == 2.0
    assert out.item(0, "available_at") == at(8)


def test_an_observation_available_exactly_at_the_cutoff_is_usable() -> None:
    """The contract is available_at <= decision_at, inclusive."""
    out = asof_join(decisions(("a", 10)), observations(("a", 10, 5.0)), as_of_col="as_of")
    assert out.item(0, "rate") == 5.0


def test_an_observation_one_microsecond_late_is_not() -> None:
    right = observations(("a", 10, 5.0)).with_columns(
        (pl.col("available_at") + pl.duration(microseconds=1)).alias("available_at")
    )
    out = asof_join(decisions(("a", 10)), right, as_of_col="as_of")
    assert out.item(0, "rate") is None


def test_keys_are_isolated() -> None:
    """Instrument b's observation must never be attached to instrument a."""
    out = asof_join(
        decisions(("a", 10), ("b", 10)),
        observations(("b", 5, 9.0)),
        as_of_col="as_of",
    ).sort("instrument_id")
    assert out.get_column("rate").to_list() == [None, 9.0]


def test_no_eligible_observation_yields_null() -> None:
    out = asof_join(decisions(("a", 3)), observations(("a", 4, 1.0)), as_of_col="as_of")
    assert out.item(0, "rate") is None
    assert out.item(0, "available_at") is None


def test_max_staleness_drops_old_matches() -> None:
    right = observations(("a", 2, 1.0))
    fresh = asof_join(decisions(("a", 10)), right, as_of_col="as_of")
    assert fresh.item(0, "rate") == 1.0
    stale = asof_join(
        decisions(("a", 10)), right, as_of_col="as_of", max_staleness=dt.timedelta(minutes=5)
    )
    assert stale.item(0, "rate") is None


def test_input_order_does_not_matter() -> None:
    left = decisions(("a", 10), ("a", 5), ("b", 7))
    right = observations(("a", 8, 2.0), ("b", 1, 7.0), ("a", 4, 1.0))
    out = asof_join(left, right, as_of_col="as_of").sort(["instrument_id", "as_of"])
    assert out.get_column("rate").to_list() == [1.0, 2.0, 7.0]


def test_the_right_side_must_carry_available_at() -> None:
    right = observations(("a", 4, 1.0)).rename({"available_at": "event_at"})
    with pytest.raises(ValueError, match="must carry an 'available_at' column"):
        asof_join(decisions(("a", 10)), right, as_of_col="as_of")


def test_a_missing_cutoff_column_is_an_error() -> None:
    with pytest.raises(ValueError, match="no cutoff column"):
        asof_join(decisions(("a", 10)), observations(("a", 4, 1.0)), as_of_col="decision_at")


def test_prefix_applies_to_payload_and_attached_availability() -> None:
    out = asof_join(
        decisions(("a", 10)),
        observations(("a", 4, 1.0)),
        as_of_col="as_of",
        right_prefix="funding_",
    )
    assert "funding_rate" in out.columns
    assert "funding_available_at" in out.columns
    assert "available_at" not in out.columns


def test_column_collisions_are_refused_not_suffixed() -> None:
    """A left frame that already has available_at (a feature frame) needs a prefix."""
    left = decisions(("a", 10)).with_columns(pl.col("as_of").alias("available_at"))
    with pytest.raises(ValueError, match="pass a right_prefix"):
        asof_join(left, observations(("a", 4, 1.0)), as_of_col="as_of")
    out = asof_join(left, observations(("a", 4, 1.0)), as_of_col="as_of", right_prefix="f_")
    assert out.item(0, "f_rate") == 1.0


def test_joining_a_feature_frame_on_its_own_availability() -> None:
    """The intended use: a feature row's cutoff is the instant that row is usable."""
    left = pl.DataFrame(
        {
            "instrument_id": ["a", "a"],
            "bar_start": [at(0), at(1)],
            "ret_1": [None, 0.01],
            "available_at": [at(1, 2), at(2, 2)],
        },
        schema_overrides={"bar_start": TS, "available_at": TS},
    )
    out = asof_join(
        left,
        observations(("a", 1, 0.5), ("a", 2, 0.7)),
        as_of_col="available_at",
        right_prefix="funding_",
    ).sort("bar_start")
    # Row at 00:01:02 sees the 00:01 observation; row at 00:02:02 sees the 00:02 one.
    assert out.get_column("funding_rate").to_list() == [0.5, 0.7]
    assert (out.get_column("funding_available_at") <= out.get_column("available_at")).all()


def test_the_postcondition_guard_fires_if_the_engine_ever_regresses(monkeypatch) -> None:
    """Simulate a broken join strategy and confirm the invariant check catches it."""
    original = pl.DataFrame.join_asof

    def forward(self, other, **kwargs):
        kwargs["strategy"] = "forward"
        return original(self, other, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "join_asof", forward)
    with pytest.raises(AvailabilityViolation, match="available after the cutoff"):
        asof_join(decisions(("a", 3)), observations(("a", 4, 1.0)), as_of_col="as_of")


def test_accepts_lazy_frames() -> None:
    out = asof_join(
        decisions(("a", 10)).lazy(), observations(("a", 4, 1.0)).lazy(), as_of_col="as_of"
    )
    assert isinstance(out, pl.DataFrame)
    assert out.item(0, "rate") == 1.0
