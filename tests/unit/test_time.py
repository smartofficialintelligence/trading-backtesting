"""The UTC boundary policy: naive rejected, other zones converted."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import BaseModel, ValidationError

from qresearch.time import (
    NaiveDatetimeError,
    UtcDatetime,
    ensure_utc,
    format_duration,
    parse_duration,
    to_epoch_micros,
)


class Boundary(BaseModel):
    at: UtcDatetime


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(NaiveDatetimeError, match="naive datetime"):
        ensure_utc(dt.datetime(2024, 1, 1, 12, 0))  # noqa: DTZ001 -- naive on purpose


def test_naive_datetime_is_rejected_at_a_model_boundary() -> None:
    with pytest.raises(ValidationError):
        Boundary(at=dt.datetime(2024, 1, 1, 12, 0))  # noqa: DTZ001 -- naive on purpose


def test_naive_iso_string_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Boundary.model_validate({"at": "2024-01-01T12:00:00"})


def test_offset_datetime_is_converted_not_relabelled() -> None:
    """A -05:00 timestamp denotes an instant; conversion must move the wall clock."""
    eastern = dt.datetime(2024, 1, 1, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    assert ensure_utc(eastern) == dt.datetime(2024, 1, 1, 17, 0, tzinfo=dt.UTC)


def test_offset_string_is_accepted_and_converted() -> None:
    assert Boundary.model_validate({"at": "2024-01-01T12:00:00-05:00"}).at == dt.datetime(
        2024, 1, 1, 17, 0, tzinfo=dt.UTC
    )


@pytest.mark.parametrize(
    ("text", "seconds"), [("1s", 1), ("1m", 60), ("5m", 300), ("1h", 3600), ("1d", 86400)]
)
def test_parse_duration(text: str, seconds: int) -> None:
    assert parse_duration(text) == dt.timedelta(seconds=seconds)


@pytest.mark.parametrize("text", ["0m", "-1m", "1", "m", "1.5m", "90", "1w", "1M", ""])
def test_parse_duration_rejects_non_canonical(text: str) -> None:
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration(text)


def test_duration_round_trips_to_the_largest_exact_unit() -> None:
    assert format_duration(dt.timedelta(seconds=300)) == "5m"
    assert format_duration(dt.timedelta(hours=2)) == "2h"
    assert format_duration(dt.timedelta(seconds=90)) == "90s"


def test_epoch_micros_is_zero_at_the_epoch() -> None:
    assert to_epoch_micros(dt.datetime(1970, 1, 1, tzinfo=dt.UTC)) == 0


def test_epoch_micros_is_offset_independent() -> None:
    """Two spellings of one instant must hash and sort identically."""
    utc = dt.datetime(2024, 6, 1, 12, 0, tzinfo=dt.UTC)
    tokyo = utc.astimezone(dt.timezone(dt.timedelta(hours=9)))
    assert to_epoch_micros(utc) == to_epoch_micros(tokyo)
