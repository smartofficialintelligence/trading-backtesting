"""Canonical serialization must be stable across key order, types, and processes."""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from decimal import Decimal

import pytest

from qresearch.ids import canonical_json, content_hash, content_hash_full


def test_mapping_order_does_not_change_the_hash() -> None:
    a = {"z": 1, "a": {"n": [1, 2], "m": True}}
    b = {"a": {"m": True, "n": [1, 2]}, "z": 1}
    assert content_hash(a) == content_hash(b)


def test_set_iteration_order_does_not_change_the_hash() -> None:
    assert content_hash({"s": {3, 1, 2}}) == content_hash({"s": {2, 3, 1}})


def test_datetime_offset_does_not_change_the_hash() -> None:
    utc = dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC)
    other = utc.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    assert content_hash({"t": utc}) == content_hash({"t": other})


def test_floats_round_trip_exactly() -> None:
    """0.1 + 0.2 must not collide with 0.3; a lossy float format would hide real diffs."""
    assert content_hash({"x": 0.1 + 0.2}) != content_hash({"x": 0.3})


def test_int_and_float_are_distinguished() -> None:
    assert content_hash({"x": 1}) != content_hash({"x": 1.0})


def test_bool_and_int_are_distinguished() -> None:
    assert content_hash({"x": True}) != content_hash({"x": 1})


def test_decimal_normalizes_trailing_zeros() -> None:
    assert content_hash({"d": Decimal("1.50")}) == content_hash({"d": Decimal("1.5")})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": value})


def test_non_string_keys_are_refused() -> None:
    with pytest.raises(TypeError, match="string keys"):
        canonical_json({1: "a"})


def test_unknown_types_are_refused_rather_than_stringified() -> None:
    with pytest.raises(TypeError, match="cannot canonicalize"):
        canonical_json({"x": object()})


def test_hash_is_stable_across_processes() -> None:
    """PYTHONHASHSEED randomizes str hashing; the content hash must not depend on it."""
    program = (
        "import sys; sys.path.insert(0, 'src');"
        "from qresearch.ids import content_hash_full;"
        "print(content_hash_full({'a': [1, 2.5, 'x'], 'b': {'c': None}}))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(outputs) == 1
    assert outputs.pop() == content_hash_full({"a": [1, 2.5, "x"], "b": {"c": None}})
