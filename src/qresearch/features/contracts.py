"""Feature definitions.

A feature is a pure Polars expression plus a :class:`FeatureSpec` describing what it
consumes and how far back it looks. The expression is deliberately all a feature author
writes; timing -- availability propagation, warm-up, contiguity across gaps -- is owned by
:mod:`qresearch.features.pipeline` and derived from the spec, so it cannot be forgotten
per feature.

The contract with the pipeline:

* The expression is evaluated ``.over("instrument_id")`` on a frame sorted by
  ``(instrument_id, bar_start)``. It may use ``shift(k)`` and rolling windows for
  ``k <= lookback_bars``. It must not use negative shifts, centred windows, or whole-
  column aggregates -- and the leakage helpers in :mod:`qresearch.features.leakage`
  exist to prove that it does not.
* ``lookback_bars`` is a promise: "this value at bar N depends on bars N-lookback..N and
  nothing else". The pipeline uses it to compute when the value becomes available and to
  null the value when those bars are not contiguous.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated, Final, Protocol, runtime_checkable

import polars as pl
from pydantic import Field, StringConstraints

from qresearch.config import FrozenModel

FEATURE_KEY: Final = ("instrument_id", "bar_start")
"""Row identity of a feature frame. Matches the bar natural key minus bar_size, which is
fixed per frame."""

FeatureName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]

ParamValue = int | float | str | bool


class FeatureSpec(FrozenModel):
    """What a feature consumes, how far back it looks, and how to identify it."""

    name: FeatureName
    """Output column name."""

    version: int = Field(default=1, ge=1)
    """Bump when the implementation changes in a way that changes output."""

    implementation: str = Field(min_length=1)
    """Fully qualified class name, so the fingerprint ties to code identity."""

    inputs: tuple[str, ...] = Field(min_length=1)
    """Bar columns the expression reads."""

    lookback_bars: int = Field(ge=0)
    """Prior bars required beyond the current one. Zero for a same-bar transform."""

    computation_latency: _dt.timedelta = Field(default=_dt.timedelta(0), ge=_dt.timedelta(0))
    """Added to availability: time to compute or deliver the value once inputs exist."""

    params: dict[str, ParamValue] = Field(default_factory=dict)

    require_contiguous: bool = True
    """Null the value when the lookback window spans a gap in the bar grid.

    A positional window over bars that are not consecutive in time is a different
    quantity from what the feature claims to measure. On by default; a feature that is
    genuinely gap-tolerant may opt out.
    """

    @property
    def window_bars(self) -> int:
        """Bars in the window including the current one."""
        return self.lookback_bars + 1

    @property
    def fingerprint(self) -> str:
        """Content hash of the spec: name, version, implementation, and parameters."""
        return self.content_hash()


@runtime_checkable
class Feature(Protocol):
    """A feature implementation."""

    @property
    def spec(self) -> FeatureSpec: ...

    def expression(self) -> pl.Expr:
        """The value expression, to be evaluated ``.over("instrument_id")``.

        Must be causal within ``spec.lookback_bars``. Returns a single unnamed expression;
        the pipeline aliases it to ``spec.name``.
        """
        ...
