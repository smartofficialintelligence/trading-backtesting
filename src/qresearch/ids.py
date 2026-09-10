"""Typed identifiers and canonical hashing.

Canonical hashing underpins two guarantees in ARCHITECTURE.md:

* a normalized dataset gets a content-addressed ``dataset_id`` (sec. 4), so re-ingesting
  identical source data under an identical policy resolves to the same dataset;
* a run gets a ``run_id`` derived from its resolved specification (sec. 6), so an
  identical experiment is recognisable as such.

Both require a serialization that is stable across mapping insertion order, across
processes, and across machines. ``json.dumps`` with ``sort_keys=True`` gets most of the
way; the rest is handled by normalizing the value types we actually persist.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from decimal import Decimal
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final, NewType
from uuid import UUID

from qresearch.time import ensure_utc

InstrumentId = NewType("InstrumentId", str)
"""Stable internal instrument identity, deliberately distinct from a provider ticker."""

DatasetId = NewType("DatasetId", str)
"""Content-addressed identity of one normalized dataset version."""

FeatureSetId = NewType("FeatureSetId", str)
RunId = NewType("RunId", str)
ExperimentId = NewType("ExperimentId", str)
OrderId = NewType("OrderId", str)

HASH_PREFIX_LENGTH: Final = 16
"""Hex characters retained in a short content hash.

64 bits of a SHA-256 is far more than enough to keep local datasets and runs distinct,
and short enough to appear in a directory name. Full digests remain available via
:func:`content_hash_full`.
"""


def _canonicalize(value: Any) -> Any:
    """Reduce a value to JSON-native types with exactly one representation each."""
    match value:
        case None | bool() | int() | str():
            return value
        case float():
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"cannot canonicalize non-finite float {value!r}")
            # repr() is the shortest string that round-trips the IEEE-754 double, and is
            # identical on every conforming platform.
            return repr(value)
        case Decimal():
            return format(value.normalize(), "f")
        case _dt.datetime():
            return ensure_utc(value).isoformat().replace("+00:00", "Z")
        case _dt.date():
            return value.isoformat()
        case _dt.timedelta():
            return f"P{value // _dt.timedelta(microseconds=1)}US"
        case UUID() | PurePosixPath():
            return str(value)
        case Enum():
            return _canonicalize(value.value)
        case dict():
            keys = [k for k in value if not isinstance(k, str)]
            if keys:
                raise TypeError(f"canonical JSON requires string keys; got {keys!r}")
            return {k: _canonicalize(v) for k, v in sorted(value.items())}
        case list() | tuple():
            return [_canonicalize(v) for v in value]
        case set() | frozenset():
            # Sort by canonical form so set contents hash independently of iteration order.
            return sorted((_canonicalize(v) for v in value), key=json.dumps)
        case _:
            dumper = getattr(value, "model_dump", None)
            if callable(dumper):
                return _canonicalize(dumper(mode="python"))
            raise TypeError(f"cannot canonicalize {type(value).__name__}: {value!r}")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to its single canonical JSON spelling."""
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash_full(value: Any) -> str:
    """Full SHA-256 hex digest of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    """Short SHA-256 hex digest (see :data:`HASH_PREFIX_LENGTH`)."""
    return content_hash_full(value)[:HASH_PREFIX_LENGTH]


def file_hash_full(path: Any, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, for manifest partition checksums."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
