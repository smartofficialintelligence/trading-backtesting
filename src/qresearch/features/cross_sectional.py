"""Cross-sectional features: values compared *across* instruments at one instant.

These are a second stage over a feature frame rather than a kind of per-instrument
feature, because their timing is different in kind. A per-instrument feature at bar N is
usable when that instrument's inputs are in. A rank at bar N is usable only when *every*
instrument's value at N is in -- the batch must be complete, or the rank is over a
different set than the one it will eventually be over. ARCHITECTURE.md sec. 9 lists
exactly this ("cross-sectional asynchrony") as a leakage mode.

Consequently a row's ``available_at`` after this stage is the maximum across the whole
cross-section at that ``bar_start``. That delays the per-instrument columns in the same
row too. This is the honest price of batch semantics; a caller who wants the
per-instrument columns earlier keeps them in a separate frame.

Missing-member policy
    Rows whose input value is null (warm-up, a nulled gap) are not members of that
    instant's cross-section. Instruments with no bar at all at that instant are likewise
    absent. The rank is computed over the members present, and is null when fewer than
    ``min_members`` are -- a rank among one is not a rank. With a static universe of N,
    ``min_members=N`` demands a full batch; the default of 2 tolerates gaps and delisting,
    at the cost of ranks whose denominator varies. Point-in-time universe membership
    (ARCHITECTURE.md sec. 4) will make that denominator explicit when it lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import polars as pl

from qresearch.features.contracts import FEATURE_KEY

BAR = FEATURE_KEY[1]
RankMethod = Literal["average", "min", "max", "dense", "ordinal"]


@dataclass(frozen=True, slots=True)
class CrossSectionalRank:
    """Rank of ``column`` among the instruments present at each ``bar_start``.

    ``pct=True`` (default) divides by the member count so the output lies in ``(0, 1]``
    and is comparable across instants with different membership.
    """

    column: str
    min_members: int = 2
    pct: bool = True
    method: RankMethod = "average"
    descending: bool = False

    def __post_init__(self) -> None:
        if self.min_members < 2:
            raise ValueError(
                f"min_members must be >= 2 (a rank among one is not a rank), got {self.min_members}"
            )

    @property
    def name(self) -> str:
        return f"{self.column}_xrank"

    def expression(self) -> pl.Expr:
        members = pl.col(self.column).count().over(BAR)  # count() excludes nulls
        rank = pl.col(self.column).rank(method=self.method, descending=self.descending).over(BAR)
        value = rank / members if self.pct else rank.cast(pl.Float64)
        return pl.when(members >= self.min_members).then(value).otherwise(None)


@dataclass(frozen=True, slots=True)
class CrossSectionalMean:
    """Equal-weight mean of ``column`` across the instruments present at each ``bar_start``.

    Every instrument's row carries the same value: a market-level reading. Over trailing
    log returns it is the return of an equal-weight basket rebalanced each bar -- the
    basket trend of D70. Membership follows the same missing-member policy as ranks.
    """

    column: str
    min_members: int = 2

    def __post_init__(self) -> None:
        if self.min_members < 2:
            raise ValueError(
                "min_members must be >= 2 (a basket of one is not a basket), "
                f"got {self.min_members}"
            )

    @property
    def name(self) -> str:
        return f"{self.column}_xmean"

    def expression(self) -> pl.Expr:
        members = pl.col(self.column).count().over(BAR)
        return pl.when(members >= self.min_members).then(pl.col(self.column).mean().over(BAR))


CrossSectionalFeature = CrossSectionalRank | CrossSectionalMean

_REQUIRED: Final = (*FEATURE_KEY, "available_at")


def compute_cross_sectional(
    frame: pl.DataFrame,
    ranks: Sequence[CrossSectionalFeature],
) -> pl.DataFrame:
    """Add cross-sectional features to a feature frame and lift its availability to batch time.

    Args:
        frame: output of :func:`~qresearch.features.pipeline.compute_features`, or any
            frame with ``instrument_id``, ``bar_start``, ``available_at`` and the input
            columns.
        ranks: the ranks and means to compute. Output names must not collide with existing
            columns.
    """
    if not ranks:
        raise ValueError("no cross-sectional features requested")
    for column in _REQUIRED:
        if column not in frame.columns:
            raise ValueError(f"frame lacks required column {column!r}")
    missing = sorted({r.column for r in ranks} - set(frame.columns))
    if missing:
        raise ValueError(f"ranked columns {missing} are not present in the frame")
    names = [r.name for r in ranks]
    if len(set(names)) != len(names):
        raise ValueError("duplicate cross-sectional feature names")
    collisions = sorted(set(names) & set(frame.columns))
    if collisions:
        raise ValueError(f"output names {collisions} collide with existing columns")
    duplicated = frame.select(list(FEATURE_KEY)).is_duplicated()
    if duplicated.any():
        raise ValueError("frame has duplicate (instrument_id, bar_start) rows")

    return (
        frame.sort(list(FEATURE_KEY))
        .with_columns(
            *[r.expression().alias(r.name) for r in ranks],
            # The batch at this instant is complete only when its slowest member is in.
            pl.col("available_at").max().over(BAR).alias("available_at"),
        )
        .select(
            *FEATURE_KEY, *[c for c in frame.columns if c not in _REQUIRED], *names, "available_at"
        )
    )
