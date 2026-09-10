"""UTC time policy for the whole system.

Policy (Stage 0, ARCHITECTURE.md sec. 2 "Point-in-time correctness is a data contract"):

* Naive datetimes are **rejected** at every boundary. A naive timestamp carries no
  information about which instant it denotes, and silently assuming UTC is exactly the
  kind of quiet mistake that produces an off-by-one-session backtest.
* Aware datetimes in a non-UTC zone are **converted** to UTC. They denote an unambiguous
  instant, so conversion is lossless for our purposes.
* The canonical stored resolution is microseconds, matching Parquet ``timestamp[us, UTC]``
  and Python's ``datetime``.

Wall-clock-with-timezone arithmetic (exchange sessions, DST) belongs in ``data.calendars``
and must produce aware datetimes before it reaches any contract.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Annotated, Any, Final

from pydantic import AfterValidator, BeforeValidator

UTC: Final = _dt.UTC

_DURATION_RE: Final = re.compile(r"^(?P<value>[1-9][0-9]*)(?P<unit>[smhd])$")
_UNIT_SECONDS: Final[dict[str, int]] = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class NaiveDatetimeError(ValueError):
    """Raised when a naive datetime reaches a system boundary."""


def now_utc() -> _dt.datetime:
    """Current instant as an aware UTC datetime."""
    return _dt.datetime.now(tz=UTC)


def ensure_utc(value: _dt.datetime) -> _dt.datetime:
    """Return ``value`` as an aware UTC datetime, rejecting naive input.

    Raises:
        NaiveDatetimeError: if ``value`` has no timezone, or a timezone whose offset is
            undefined for that instant.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise NaiveDatetimeError(
            f"naive datetime {value!r} is not allowed; attach a timezone "
            "(all system boundaries require aware UTC timestamps)"
        )
    return value.astimezone(UTC)


def _coerce_utc(value: Any) -> Any:
    """Pydantic BeforeValidator: reject naive input before pydantic can coerce it."""
    if isinstance(value, _dt.datetime):
        return ensure_utc(value)
    if isinstance(value, str):
        # Let pydantic parse the string, but pre-reject anything without an offset so the
        # error names the real problem rather than a downstream type mismatch.
        parsed = _dt.datetime.fromisoformat(value)
        return ensure_utc(parsed)
    return value


def _assert_utc(value: _dt.datetime) -> _dt.datetime:
    return ensure_utc(value)


UtcDatetime = Annotated[_dt.datetime, BeforeValidator(_coerce_utc), AfterValidator(_assert_utc)]
"""An aware UTC datetime. Naive values raise; other zones convert."""


def parse_duration(text: str) -> _dt.timedelta:
    """Parse a canonical duration such as ``1m``, ``5m``, ``1h``, ``1d``.

    Only whole positive multiples of a single unit are canonical, so that a duration has
    exactly one spelling and ``bar_size`` values compare as strings.
    """
    match = _DURATION_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"invalid duration {text!r}; expected a positive integer followed by "
            "one of s, m, h, d (for example '1m', '5m', '1h')"
        )
    value = int(match["value"])
    return _dt.timedelta(seconds=value * _UNIT_SECONDS[match["unit"]])


def format_duration(delta: _dt.timedelta) -> str:
    """Render a timedelta in the canonical spelling used by :func:`parse_duration`.

    Chooses the largest unit that divides the duration exactly, so 300s renders as ``5m``.
    """
    total = delta // _dt.timedelta(seconds=1)
    if delta != _dt.timedelta(seconds=total) or total <= 0:
        raise ValueError(f"duration {delta!r} is not a positive whole number of seconds")
    for unit in ("d", "h", "m", "s"):
        size = _UNIT_SECONDS[unit]
        if total % size == 0:
            return f"{total // size}{unit}"
    raise AssertionError("unreachable: seconds divides every integer")


def as_utc_scalar(value: Any) -> _dt.datetime:
    """Narrow a dataframe scalar to an aware UTC datetime.

    Polars aggregations are typed as a broad scalar union. Rather than suppressing that at
    each call site, this asserts the type once and fails loudly if a column that should
    hold timestamps does not.
    """
    if not isinstance(value, _dt.datetime):
        raise TypeError(
            f"expected a datetime scalar, got {type(value).__name__}: {value!r}; the "
            "column is not a timestamp column"
        )
    return ensure_utc(value)


def to_epoch_micros(value: _dt.datetime) -> int:
    """Microseconds since the Unix epoch, for canonical hashing and stable sorting."""
    aware = ensure_utc(value)
    return (aware - _dt.datetime(1970, 1, 1, tzinfo=UTC)) // _dt.timedelta(microseconds=1)
