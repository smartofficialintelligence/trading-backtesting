"""Pydantic base models shared by every contract and configuration object.

The settings here are the enforcement half of the "point-in-time correctness is a data
contract" principle. Every model in the system inherits one of these bases so that the
guarantees are structural rather than remembered:

* ``extra="forbid"`` — a misspelled config key is an error, not a silently ignored
  parameter that leaves the default economic assumption in place.
* ``strict=True`` — no ``"1.5"`` becoming ``1.5``, no ``1`` becoming ``True``. Loose
  coercion is how a string timestamp column quietly becomes something else.
* ``frozen=True`` on :class:`FrozenModel` — contracts that are hashed or persisted must
  not mutate after validation.
* ``validate_default=True`` — defaults are held to the same rules as supplied values.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from qresearch.ids import canonical_json, content_hash


class StrictModel(BaseModel):
    """Mutable strict base for configuration objects still being assembled."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )

    def canonical_dict(self) -> dict[str, Any]:
        """Model contents in the form used for canonical hashing."""
        return self.model_dump(mode="python")

    def canonical_json(self) -> str:
        """The single canonical JSON spelling of this model."""
        return canonical_json(self.canonical_dict())

    def content_hash(self) -> str:
        """Short content hash, stable across key order, processes, and machines."""
        return content_hash(self.canonical_dict())


class FrozenModel(StrictModel):
    """Immutable strict base for persisted contracts and identity-bearing records."""

    model_config = ConfigDict(frozen=True)
