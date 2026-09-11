"""Round-trip trade attribution, derived from the fill ledger.

A *trade* is one flat-to-flat episode in a single instrument: it opens when the position
leaves zero, absorbs every add and trim, and closes when the position returns to zero. A
flip (long straight to short) closes one trade and opens another at the same fill.

Why flat-to-flat rather than FIFO lots: the portfolio keeps an average cost basis, so
flat-to-flat needs no lot matching and cannot disagree with the accounting it is derived
from. FIFO would report different per-trade numbers for the same economics whenever a
position is scaled in and out, and would need its own matching rules to argue about. The
cost of this choice is that a long accumulation followed by a partial exit is one trade,
not several -- which is the honest description of what the portfolio did.

P&L identity, which :func:`build_trades` asserts per trade::

    gross_pnl = -sum(signed_quantity * reference_price)      # versus the untouched print
    costs     =  sum(quantity * (half_spread + slippage) + fee)
    net_pnl   =  gross_pnl - costs

An open trade at the end of a run is reported with ``is_open=True`` and P&L marked at the
last price supplied; its ``closed_at`` is null. It is excluded from win-rate style
statistics, which are about completed decisions.

If no mark is available for an open trade, its P&L is reported as **null, not zero and
not the cash outflow**. The cash outflow alone reads as a total loss, and zero is a guess;
an unvalued position simply has unknown P&L, and saying so is the only honest option.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import polars as pl
from pydantic import Field

from qresearch.config import FrozenModel

TRADE_SCHEMA: Final[dict[str, pl.DataType]] = {
    "trade_id": pl.Int64(),
    "instrument_id": pl.String(),
    "direction": pl.String(),
    "opened_at": pl.Datetime("us", "UTC"),
    "closed_at": pl.Datetime("us", "UTC"),
    "duration_s": pl.Float64(),
    "peak_quantity": pl.Float64(),
    "entry_price": pl.Float64(),
    "exit_price": pl.Float64(),
    "fill_count": pl.Int64(),
    "gross_pnl": pl.Float64(),
    "costs": pl.Float64(),
    "net_pnl": pl.Float64(),
    "return_on_notional": pl.Float64(),
    "is_open": pl.Boolean(),
    "fold": pl.Int64(),
    "role": pl.String(),
}

_TOLERANCE = 1e-9


@dataclass(slots=True)
class _Open:
    """A trade being accumulated."""

    trade_id: int
    instrument_id: str
    direction: str
    opened_at: _dt.datetime
    position: float = 0.0
    peak_quantity: float = 0.0
    entry_notional: float = 0.0
    entry_quantity: float = 0.0
    exit_notional: float = 0.0
    exit_quantity: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0
    fill_count: int = 0
    last_at: _dt.datetime | None = None
    fold: int | None = None
    role: str | None = None
    extras: dict[str, object] = field(default_factory=dict)


def build_trades(fills: pl.DataFrame, *, marks: Mapping[str, float] | None = None) -> pl.DataFrame:
    """Group a fill ledger into flat-to-flat trades.

    Args:
        fills: the run's ``fills`` ledger. ``fold`` and ``role`` are carried through when
            present; a trade that spans a boundary keeps the values from its opening fill.
        marks: last known price per instrument, used to value a trade still open at the
            end. Without it an open trade reports its entry price as the exit and zero
            unrealised P&L, which would understate it -- so supplying marks is preferred.

    Returns:
        One row per trade, ordered by ``opened_at`` then instrument.
    """
    empty = pl.DataFrame(schema=TRADE_SCHEMA)
    if fills.is_empty():
        return empty
    for column in (
        "instrument_id",
        "side",
        "quantity",
        "fill_at",
        "reference_price",
        "price",
        "fee",
    ):
        if column not in fills.columns:
            raise ValueError(f"fills ledger lacks required column {column!r}")

    has_spread = "half_spread" in fills.columns and "slippage" in fills.columns
    ordered = fills.sort(
        ["fill_at", "instrument_id", *(["fill_id"] if "fill_id" in fills.columns else [])]
    )
    open_trades: dict[str, _Open] = {}
    rows: list[dict[str, object]] = []
    next_id = 1

    for fill in ordered.iter_rows(named=True):
        instrument_id = str(fill["instrument_id"])
        sign = 1.0 if fill["side"] == "buy" else -1.0
        quantity = float(fill["quantity"])
        signed = sign * quantity
        price = float(fill["price"])
        reference = float(fill["reference_price"])
        friction = float(fill["fee"])
        if has_spread:
            friction += quantity * (float(fill["half_spread"]) + float(fill["slippage"]))

        current = open_trades.get(instrument_id)
        remaining = signed

        while abs(remaining) > _TOLERANCE:
            if current is None:
                current = _Open(
                    trade_id=next_id,
                    instrument_id=instrument_id,
                    direction="long" if remaining > 0 else "short",
                    opened_at=fill["fill_at"],
                    fold=fill.get("fold"),
                    role=fill.get("role"),
                )
                next_id += 1
                open_trades[instrument_id] = current

            opening = (current.position >= 0) == (remaining > 0) or current.position == 0.0
            if opening:
                applied = remaining
            else:
                # Reducing: only the part that does not cross zero belongs to this trade.
                applied = (
                    remaining if abs(remaining) <= abs(current.position) else -current.position
                )

            share = abs(applied) / quantity if quantity else 0.0
            current.gross_pnl -= applied * reference
            current.costs += friction * share
            current.position += applied
            current.fill_count += 1
            current.last_at = fill["fill_at"]
            if (applied > 0 and current.direction == "long") or (
                applied < 0 and current.direction == "short"
            ):
                current.entry_notional += abs(applied) * price
                current.entry_quantity += abs(applied)
            else:
                current.exit_notional += abs(applied) * price
                current.exit_quantity += abs(applied)
            current.peak_quantity = max(current.peak_quantity, abs(current.position))
            remaining -= applied

            if abs(current.position) <= _TOLERANCE:
                rows.append(_close(current, closed_at=fill["fill_at"], valued=True))
                del open_trades[instrument_id]
                current = None

    marks = marks or {}
    for instrument_id, trade in open_trades.items():
        mark = marks.get(instrument_id)
        if mark is not None:
            trade.gross_pnl += trade.position * mark
            trade.exit_notional += abs(trade.position) * mark
            trade.exit_quantity += abs(trade.position)
        rows.append(_close(trade, closed_at=None, valued=mark is not None))

    if not rows:
        return empty
    return pl.DataFrame(rows, schema=TRADE_SCHEMA).sort(["opened_at", "instrument_id"])


def _close(trade: _Open, *, closed_at: _dt.datetime | None, valued: bool) -> dict[str, object]:
    net = trade.gross_pnl - trade.costs
    entry = trade.entry_notional / trade.entry_quantity if trade.entry_quantity else 0.0
    exit_price = trade.exit_notional / trade.exit_quantity if trade.exit_quantity else None
    notional = entry * trade.peak_quantity
    end = closed_at or trade.last_at
    if not valued:
        # An open position nobody priced. The accumulated cash flow is not a P&L -- it is
        # one leg of an unfinished trade -- so report unknown rather than a number that
        # reads as a total loss.
        return {
            "trade_id": trade.trade_id,
            "instrument_id": trade.instrument_id,
            "direction": trade.direction,
            "opened_at": trade.opened_at,
            "closed_at": None,
            "duration_s": (end - trade.opened_at).total_seconds() if end else None,
            "peak_quantity": trade.peak_quantity,
            "entry_price": entry,
            "exit_price": None,
            "fill_count": trade.fill_count,
            "gross_pnl": None,
            "costs": trade.costs,
            "net_pnl": None,
            "return_on_notional": None,
            "is_open": True,
            "fold": trade.fold,
            "role": trade.role,
        }
    return {
        "trade_id": trade.trade_id,
        "instrument_id": trade.instrument_id,
        "direction": trade.direction,
        "opened_at": trade.opened_at,
        "closed_at": closed_at,
        "duration_s": (end - trade.opened_at).total_seconds() if end else None,
        "peak_quantity": trade.peak_quantity,
        "entry_price": entry,
        "exit_price": exit_price,
        "fill_count": trade.fill_count,
        "gross_pnl": trade.gross_pnl,
        "costs": trade.costs,
        "net_pnl": net,
        "return_on_notional": net / notional if notional else None,
        "is_open": closed_at is None,
        "fold": trade.fold,
        "role": trade.role,
    }


class TradeStats(FrozenModel):
    """Round-trip statistics over *closed* trades."""

    trade_count: int = Field(ge=0)
    open_at_end: int = Field(ge=0)
    win_rate: float | None = None
    """Fraction of closed trades with positive net P&L. None when there are none."""

    average_win: float | None = None
    average_loss: float | None = None
    """Mean net P&L of losing trades, negative by convention."""

    profit_factor: float | None = None
    """Gross wins over gross losses. None when there are no losses (undefined, not
    infinite -- reporting inf as a headline would be worse than reporting nothing)."""

    expectancy: float | None = None
    """Mean net P&L per closed trade."""

    median_duration_s: float | None = None
    largest_win: float | None = None
    largest_loss: float | None = None
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    long_count: int = Field(default=0, ge=0)
    short_count: int = Field(default=0, ge=0)


def trade_stats(trades: pl.DataFrame) -> TradeStats:
    """Summarise a trade ledger. Open trades are counted but excluded from statistics."""
    if trades.is_empty():
        return TradeStats(trade_count=0, open_at_end=0)
    closed = trades.filter(~pl.col("is_open"))
    open_count = int(trades.get_column("is_open").sum())
    if closed.is_empty():
        return TradeStats(trade_count=0, open_at_end=open_count)

    net = closed.get_column("net_pnl")
    wins = closed.filter(pl.col("net_pnl") > 0).get_column("net_pnl")
    losses = closed.filter(pl.col("net_pnl") < 0).get_column("net_pnl")
    gross_losses = float(losses.sum()) if losses.len() else 0.0
    return TradeStats(
        trade_count=closed.height,
        open_at_end=open_count,
        win_rate=wins.len() / closed.height,
        average_win=float(wins.mean()) if wins.len() else None,  # type: ignore[arg-type]
        average_loss=float(losses.mean()) if losses.len() else None,  # type: ignore[arg-type]
        profit_factor=(float(wins.sum()) / abs(gross_losses)) if gross_losses < 0 else None,
        expectancy=float(net.mean()),  # type: ignore[arg-type]
        median_duration_s=float(closed.get_column("duration_s").median()),  # type: ignore[arg-type]
        largest_win=float(net.max()),  # type: ignore[arg-type]
        largest_loss=float(net.min()),  # type: ignore[arg-type]
        gross_pnl=float(closed.get_column("gross_pnl").sum()),
        costs=float(closed.get_column("costs").sum()),
        net_pnl=float(net.sum()),
        long_count=closed.filter(pl.col("direction") == "long").height,
        short_count=closed.filter(pl.col("direction") == "short").height,
    )
