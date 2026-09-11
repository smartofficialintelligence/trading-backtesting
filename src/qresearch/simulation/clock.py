"""The simulation timeline: every instant at which something can happen, in order.

Two kinds of market event come out of a bar frame:

* an **open event** at ``bar_start``, carrying only the open price -- the moment an
  eligible order may fill;
* a **publication event** at ``available_at``, carrying the whole bar -- the moment a
  strategy may see it.

Feature rows publish at their own ``available_at``. Instants are the sorted union, so the
engine steps through time by *what became knowable or tradeable*, never by row index.
Within an instant, events are ordered by instrument id for stable bookkeeping; the phases
in :mod:`qresearch.simulation.engine` make that order economically irrelevant.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from qresearch.ids import InstrumentId
from qresearch.simulation.execution import OpenEvent


@dataclass(frozen=True, slots=True)
class Timeline:
    instants: list[_dt.datetime]
    opens: dict[_dt.datetime, list[OpenEvent]] = field(default_factory=dict)
    bar_publications: dict[_dt.datetime, list[dict[str, Any]]] = field(default_factory=dict)
    feature_publications: dict[_dt.datetime, list[dict[str, Any]]] = field(default_factory=dict)
    bar_size: str = ""


def build_timeline(bars: pl.DataFrame, features: pl.DataFrame | None = None) -> Timeline:
    if bars.is_empty():
        raise ValueError("cannot build a timeline from an empty bar frame")
    sizes = bars.get_column("bar_size").unique().to_list()
    if len(sizes) != 1:
        raise ValueError(f"bars must have exactly one bar_size, found {sorted(sizes)}")
    ordered = bars.sort(["bar_start", "instrument_id"])
    opens: dict[_dt.datetime, list[OpenEvent]] = defaultdict(list)
    for row in ordered.select("instrument_id", "bar_start", "open").iter_rows():
        instrument_id, bar_start, open_price = row
        opens[bar_start].append(
            OpenEvent(
                instrument_id=InstrumentId(instrument_id),
                at=bar_start,
                open=open_price,
                bar_start=bar_start,
            )
        )

    publications: dict[_dt.datetime, list[dict[str, Any]]] = defaultdict(list)
    for bar in bars.sort(["available_at", "instrument_id"]).iter_rows(named=True):
        publications[bar["available_at"]].append(bar)

    feature_publications: dict[_dt.datetime, list[dict[str, Any]]] = defaultdict(list)
    if features is not None:
        for column in ("instrument_id", "bar_start", "available_at"):
            if column not in features.columns:
                raise ValueError(f"feature frame lacks {column!r}")
        for feature_row in features.sort(["available_at", "instrument_id"]).iter_rows(named=True):
            feature_publications[feature_row["available_at"]].append(feature_row)

    instants = sorted(set(opens) | set(publications) | set(feature_publications))
    return Timeline(
        instants=instants,
        opens=dict(opens),
        bar_publications=dict(publications),
        feature_publications=dict(feature_publications),
        bar_size=sizes[0],
    )
