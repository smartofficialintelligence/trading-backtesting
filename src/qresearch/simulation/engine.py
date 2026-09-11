"""The chronological simulator.

For each instant on the :class:`~qresearch.simulation.clock.Timeline`, the phases of
ARCHITECTURE.md sec. 5 run in a fixed order:

1. expire orders whose ``expires_at`` has passed;
2. process open events: fill orders that were eligible *before this instant's decision*;
3. apply fills to the portfolio;
4. publish bars and feature rows whose ``available_at`` has arrived; update marks;
5. if anything published and the instant is inside the decision range, mark the
   portfolio and invoke the strategy once with the complete batch;
6. size and constrain the intents, schedule orders with explicit latency;
7. record a snapshot.

An order created in phase 6 cannot be seen by phase 2 of the same instant, so a decision
made on bar N (published no earlier than bar N+1's open) fills at open(N+2) at the
earliest under the default fill rule. See ``FillRule`` for the labelled optimistic
alternative.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import statistics
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from types import NoneType, UnionType
from typing import Annotated, Any, Union, get_args, get_origin

import polars as pl
from pydantic import Field

from qresearch.config import FrozenModel
from qresearch.data.contracts import Instrument
from qresearch.ids import InstrumentId, OrderId
from qresearch.research.splits import TimeRange
from qresearch.simulation.clock import Timeline, build_timeline
from qresearch.simulation.constraints import ConstraintConfig, SizingState, size_intents
from qresearch.simulation.events import (
    Fill,
    IntentOutcome,
    Order,
    OrderEvent,
    OrderIntent,
    OrderSide,
    OrderStatus,
    PortfolioSnapshot,
    PositionSnapshot,
    WarningRecord,
)
from qresearch.simulation.execution import (
    BarOpenExecutionModel,
    ExecutionConfig,
    FillRule,
    LiveOrder,
    OpenEvent,
)
from qresearch.simulation.portfolio import Portfolio
from qresearch.strategy.contracts import DecisionContext, Strategy
from qresearch.time import parse_duration


class EndOfRunPolicy(StrEnum):
    MARK = "mark"
    """Leave positions open and valued at the last point-in-time mark."""

    LIQUIDATE = "liquidate"
    """Submit flattening orders at the first instant past the decision range and keep
    processing opens until they fill or expire."""


class SimulationConfig(FrozenModel):
    initial_cash: float = Field(default=100_000.0, gt=0)
    execution: ExecutionConfig = ExecutionConfig()
    constraints: ConstraintConfig = ConstraintConfig()
    context_lookback_bars: int = Field(default=1, ge=1)
    """Recent published bars/feature rows per instrument exposed to the strategy."""

    max_mark_staleness: _dt.timedelta | None = _dt.timedelta(minutes=10)
    end_of_run: EndOfRunPolicy = EndOfRunPolicy.MARK


@dataclass(slots=True)
class SimulationResult:
    orders: list[Order]
    order_events: list[OrderEvent]
    fills: list[Fill]
    outcomes: list[IntentOutcome]
    snapshots: list[PortfolioSnapshot]
    positions: list[PositionSnapshot]
    warnings: list[WarningRecord]
    decisions: int
    final: PortfolioSnapshot
    config: SimulationConfig
    strategy_id: str
    decision_range: TimeRange

    def frames(self) -> dict[str, pl.DataFrame]:
        """Every ledger as a Polars frame, for persistence and analysis."""

        def frame(rows: list[Any], model: type[FrozenModel]) -> pl.DataFrame:
            schema = _schema_for(model)
            return pl.DataFrame([r.model_dump(mode="python") for r in rows], schema=schema)

        return {
            "orders": frame(self.orders, Order),
            "order_events": frame(self.order_events, OrderEvent),
            "fills": frame(self.fills, Fill),
            "intent_outcomes": frame(self.outcomes, IntentOutcome),
            "equity_curve": frame(self.snapshots, PortfolioSnapshot),
            "positions": frame(self.positions, PositionSnapshot),
            "warnings": frame(self.warnings, WarningRecord),
        }

    @property
    def total_costs(self) -> float:
        return sum(f.total_cost for f in self.fills)


def _schema_for(model: type[FrozenModel]) -> dict[str, pl.DataType]:
    """Polars schema from a pydantic model, so empty ledgers still have typed columns."""
    return {name: _polars_dtype(info.annotation) for name, info in model.model_fields.items()}


def _polars_dtype(annotation: Any) -> pl.DataType:
    """Resolve ``Optional``, ``Annotated``, ``NewType`` and enums to a Polars dtype.

    Getting this wrong is silent: an optional float persisted as a string still round
    trips through Parquet, it just stops being a number.
    """
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _polars_dtype(args[0])
    if origin is Union or origin is UnionType:
        remaining = [a for a in args if a is not NoneType]
        return _polars_dtype(remaining[0]) if len(remaining) == 1 else pl.String()
    supertype = getattr(annotation, "__supertype__", None)
    if supertype is not None:
        return _polars_dtype(supertype)
    if annotation is bool:
        return pl.Boolean()
    if annotation is int:
        return pl.Int64()
    if annotation is float:
        return pl.Float64()
    if annotation is _dt.datetime:
        return pl.Datetime("us", "UTC")
    if annotation is _dt.timedelta:
        return pl.Duration("us")
    return pl.String()


@dataclass(slots=True)
class _State:
    portfolio: Portfolio
    pending: list[LiveOrder] = field(default_factory=list)
    published_bars: dict[InstrumentId, deque[dict[str, Any]]] = field(default_factory=dict)
    published_features: dict[InstrumentId, deque[dict[str, Any]]] = field(default_factory=dict)
    orders: list[Order] = field(default_factory=list)
    order_events: list[OrderEvent] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    outcomes: list[IntentOutcome] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    positions: list[PositionSnapshot] = field(default_factory=list)
    warnings: dict[tuple[str, InstrumentId | None], WarningRecord] = field(default_factory=dict)
    decisions: int = 0
    next_order: int = 1
    next_fill: int = 1
    liquidated: bool = False
    open_index: dict[InstrumentId, list[OpenEvent]] = field(default_factory=dict)
    models: dict[InstrumentId, BarOpenExecutionModel] = field(default_factory=dict)
    bar_step: _dt.timedelta = _dt.timedelta(0)


def run_simulation(
    bars: pl.DataFrame,
    *,
    strategy: Strategy,
    instruments: Mapping[str, Instrument],
    config: SimulationConfig,
    decision_range: TimeRange,
    features: pl.DataFrame | None = None,
    run_id: str | None = None,
    fold: int | None = None,
) -> SimulationResult:
    """Run one deterministic simulation and return its ledgers.

    Args:
        bars: canonical bars for every instrument in the universe, one ``bar_size``. May
            extend before ``decision_range`` (warm-up) and after it (liquidation).
        strategy: invoked once per publication batch inside ``decision_range``.
        instruments: definitions keyed by ``instrument_id``; every id in ``bars`` must be
            present (for quantity increments).
        features: optional feature frame with ``available_at``; rows publish at that time.
    """
    timeline = build_timeline(bars, features)
    bar_step = parse_duration(timeline.bar_size)
    known = {InstrumentId(i) for i in instruments}
    present = {InstrumentId(i) for i in bars.get_column("instrument_id").unique().to_list()}
    if unknown := sorted(present - known):
        raise ValueError(f"bars contain instruments without definitions: {unknown}")

    increments = {
        InstrumentId(i): float(inst.quantity_increment) for i, inst in instruments.items()
    }
    state = _State(portfolio=Portfolio(initial_cash=config.initial_cash))
    lookback = max(config.context_lookback_bars, config.execution.liquidity_lookback_bars)
    for instrument_id in present:
        state.published_bars[instrument_id] = deque(maxlen=lookback)
        state.published_features[instrument_id] = deque(maxlen=config.context_lookback_bars)
    models = {
        i: BarOpenExecutionModel(config.execution, quantity_increment=increments[i])
        for i in present
    }
    state.models = models
    state.bar_step = bar_step
    for events in timeline.opens.values():
        for event in events:
            state.open_index.setdefault(event.instrument_id, []).append(event)
    for events_list in state.open_index.values():
        events_list.sort(key=lambda e: e.bar_start)
    if config.execution.fill_rule is FillRule.OPEN_OF_CURRENT_BAR:
        _warn(
            state,
            timeline.instants[0],
            "optimistic_fill_rule",
            "fill_rule=open_of_current_bar uses a print from before the order existed; "
            "results are optimistic by up to one bar of latency",
        )

    strategy.reset()
    last_instant = timeline.instants[0]
    for t in timeline.instants:
        last_instant = t
        filled = _phase_expire_and_fill(t, timeline, state, models, config, bar_step)
        published = _phase_publish(t, timeline, state, config)
        if published and decision_range.contains(t):
            _phase_decide(t, state, strategy, instruments, increments, config, run_id, fold)
        elif (
            config.end_of_run is EndOfRunPolicy.LIQUIDATE
            and not state.liquidated
            and t >= decision_range.end
        ):
            _liquidate(t, state, increments, config)
        if filled or published:
            _record_snapshot(t, state, config)
        if state.liquidated and not state.pending and t >= decision_range.end:
            break

    if config.end_of_run is EndOfRunPolicy.LIQUIDATE and not state.liquidated:
        _warn(
            state,
            last_instant,
            "liquidation_impossible",
            "no instant at or after the decision range end; positions left open and marked",
        )
    if state.pending:
        for live in state.pending:
            _order_event(
                state,
                live.order_id,
                last_instant,
                OrderStatus.CANCELLED,
                live.remaining,
                "simulation ended with the order unfilled",
            )
        state.pending.clear()
    state.portfolio.reconcile()
    final = state.portfolio.snapshot(last_instant)
    if not state.snapshots or state.snapshots[-1].at != last_instant:
        _record_snapshot(last_instant, state, config)
    return SimulationResult(
        orders=state.orders,
        order_events=state.order_events,
        fills=state.fills,
        outcomes=state.outcomes,
        snapshots=state.snapshots,
        positions=state.positions,
        warnings=sorted(
            state.warnings.values(), key=lambda w: (w.at, w.code, w.instrument_id or "")
        ),
        decisions=state.decisions,
        final=final,
        config=config,
        strategy_id=strategy.strategy_id,
        decision_range=decision_range,
    )


# -- phases ----------------------------------------------------------------------------------


def _phase_expire_and_fill(
    t: _dt.datetime,
    timeline: Timeline,
    state: _State,
    models: Mapping[InstrumentId, BarOpenExecutionModel],
    config: SimulationConfig,
    bar_step: _dt.timedelta,
) -> bool:
    still_pending: list[LiveOrder] = []
    for live in state.pending:
        expires = live.order.expires_at
        if expires is not None and expires <= t:
            _order_event(
                state,
                live.order_id,
                t,
                OrderStatus.EXPIRED,
                live.remaining,
                f"unfilled {live.remaining:.10g} expired",
            )
        else:
            still_pending.append(live)
    state.pending = still_pending

    filled_any = False
    for event in timeline.opens.get(t, ()):
        candidates = [
            live
            for live in state.pending
            if live.order.instrument_id == event.instrument_id
            and _eligible(live.order, event, config, bar_step)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda live: (live.order.eligible_at, live.order_id))
        result, state.next_fill = models[event.instrument_id].match(
            event, candidates, next_fill_id=state.next_fill
        )
        for code, message in result.notes:
            _warn(state, t, code, message, event.instrument_id)
        if _apply_fills(t, state, candidates, result.fills):
            filled_any = True
    return filled_any


def _eligible(
    order: Order, event: OpenEvent, config: SimulationConfig, bar_step: _dt.timedelta
) -> bool:
    return order.eligible_at <= event.at


def _containing_open(
    state: _State, instrument_id: InstrumentId, at: _dt.datetime
) -> OpenEvent | None:
    """The open event of the bar whose interval contains ``at``, if such a bar exists."""
    events = state.open_index.get(instrument_id, [])
    index = bisect.bisect_right([e.bar_start for e in events], at) - 1
    if index < 0:
        return None
    event = events[index]
    return event if event.bar_start <= at < event.bar_start + state.bar_step else None


def _apply_fills(
    t: _dt.datetime, state: _State, candidates: list[LiveOrder], fills: list[Fill]
) -> bool:
    filled_any = False
    for fill in fills:
        _assert_chain(fill, state)
        if fill.instrument_id not in state.portfolio.marks:
            # Filled before this instrument ever published: value at the print we
            # traded on until a close arrives. See docs/decisions.md.
            state.portfolio.set_mark(fill.instrument_id, fill.reference_price, t)
        state.portfolio.apply_fill(fill)
        state.fills.append(fill)
        filled_any = True
        live = next(c for c in candidates if c.order_id == fill.order_id)
        status = OrderStatus.FILLED if live.remaining <= 1e-12 else OrderStatus.PARTIALLY_FILLED
        _order_event(state, fill.order_id, fill.fill_at, status, fill.quantity)
    state.pending = [live for live in state.pending if live.remaining > 1e-12]
    return filled_any


def _phase_publish(
    t: _dt.datetime, timeline: Timeline, state: _State, config: SimulationConfig
) -> bool:
    published = False
    for row in timeline.bar_publications.get(t, ()):
        instrument_id = InstrumentId(row["instrument_id"])
        state.published_bars[instrument_id].append(row)
        state.portfolio.set_mark(instrument_id, float(row["close"]), t)
        published = True
    for row in timeline.feature_publications.get(t, ()):
        instrument_id = InstrumentId(row["instrument_id"])
        state.published_features.setdefault(
            instrument_id, deque(maxlen=config.context_lookback_bars)
        ).append(row)
        published = True
    return published


def _phase_decide(
    t: _dt.datetime,
    state: _State,
    strategy: Strategy,
    instruments: Mapping[str, Instrument],
    increments: Mapping[InstrumentId, float],
    config: SimulationConfig,
    run_id: str | None,
    fold: int | None,
) -> None:
    _check_staleness(t, state, config)
    context = _build_context(t, state, config, run_id, fold)
    intents = list(strategy.on_decision(context))
    state.decisions += 1
    if not intents:
        return
    _submit(t, intents, state, increments, config)


def _submit(
    t: _dt.datetime,
    intents: list[OrderIntent],
    state: _State,
    increments: Mapping[InstrumentId, float],
    config: SimulationConfig,
) -> None:
    portfolio = state.portfolio
    pending_signed: dict[InstrumentId, float] = defaultdict(float)
    for live in state.pending:
        pending_signed[live.order.instrument_id] += live.order.side.sign * live.remaining
    sizing = SizingState(
        equity=portfolio.equity,
        cash=portfolio.cash,
        marks={i: m.price for i, m in portfolio.marks.items()},
        positions={i: p.quantity for i, p in portfolio.positions.items() if not p.is_flat},
        pending=dict(pending_signed),
        increments=increments,
    )
    # Canonical order so that a strategy emitting [B, A] and one emitting [A, B] produce
    # identical ledgers, order ids included.
    intents = sorted(intents, key=lambda i: (i.instrument_id, i.kind.value, i.value, i.tag or ""))
    outcomes = size_intents(intents, sizing, config.constraints, at=t)
    exec_cfg = config.execution
    immediate = exec_cfg.fill_rule is FillRule.OPEN_OF_CURRENT_BAR
    for intent, outcome in zip(intents, outcomes, strict=True):
        if outcome.final_quantity == 0.0:
            state.outcomes.append(outcome)
            continue
        liquidity, volatility = _estimates(
            state, intent.instrument_id, exec_cfg.liquidity_lookback_bars
        )
        order_at = t + exec_cfg.submission_latency
        order = Order(
            order_id=OrderId(f"o{state.next_order:08d}"),
            instrument_id=intent.instrument_id,
            side=OrderSide.BUY if outcome.final_quantity > 0 else OrderSide.SELL,
            quantity=abs(outcome.final_quantity),
            signal_at=t,
            order_at=order_at,
            eligible_at=order_at + exec_cfg.order_latency,
            expires_at=None if exec_cfg.expire_after is None else order_at + exec_cfg.expire_after,
            liquidity_estimate=liquidity,
            volatility_estimate=volatility,
            intent_kind=intent.kind,
            tag=intent.tag,
        )
        state.next_order += 1
        state.orders.append(order)
        live = LiveOrder(order=order, remaining=order.quantity)
        state.pending.append(live)
        state.outcomes.append(outcome.model_copy(update={"order_id": order.order_id}))
        _order_event(state, order.order_id, order_at, OrderStatus.SUBMITTED, order.quantity)
        if immediate:
            containing = _containing_open(state, order.instrument_id, order.eligible_at)
            if containing is not None:
                event = OpenEvent(
                    instrument_id=order.instrument_id,
                    at=order.eligible_at,
                    open=containing.open,
                    bar_start=containing.bar_start,
                )
                result, state.next_fill = state.models[order.instrument_id].match(
                    event, [live], next_fill_id=state.next_fill
                )
                for code, message in result.notes:
                    _warn(state, t, code, message, order.instrument_id)
                _apply_fills(order.eligible_at, state, [live], result.fills)
        if (
            order.side is OrderSide.SELL
            and sizing.base(intent.instrument_id) - order.quantity < -1e-12
        ):
            _warn(
                state,
                t,
                "short_position",
                "short positions carry no borrow cost or availability constraint in this model",
                intent.instrument_id,
            )


def _liquidate(
    t: _dt.datetime,
    state: _State,
    increments: Mapping[InstrumentId, float],
    config: SimulationConfig,
) -> None:
    state.liquidated = True
    intents = [
        OrderIntent.target_quantity(i, 0.0, at=t, tag="liquidation")
        for i, p in sorted(state.portfolio.positions.items())
        if not p.is_flat
    ]
    for live in state.pending:
        _order_event(
            state,
            live.order_id,
            t,
            OrderStatus.CANCELLED,
            live.remaining,
            "cancelled for liquidation",
        )
    state.pending.clear()
    if intents:
        _submit(t, intents, state, increments, config)


def _estimates(
    state: _State, instrument_id: InstrumentId, lookback: int
) -> tuple[float | None, float | None]:
    rows = list(state.published_bars.get(instrument_id, ()))[-lookback:]
    if len(rows) < lookback:
        return None, None
    volumes = [float(r["volume"]) for r in rows]
    liquidity = sum(volumes) / len(volumes)
    closes = [float(r["close"]) for r in rows]
    returns = [b / a - 1.0 for a, b in pairwise(closes) if a > 0]
    volatility = statistics.stdev(returns) if len(returns) >= 2 else None
    return (liquidity if liquidity > 0 else None), volatility


def _build_context(
    t: _dt.datetime, state: _State, config: SimulationConfig, run_id: str | None, fold: int | None
) -> DecisionContext:
    latest_rows = [d[-1] for d in state.published_bars.values() if d]
    recent_rows = [
        r for d in state.published_bars.values() for r in list(d)[-config.context_lookback_bars :]
    ]
    bars_latest = pl.DataFrame(latest_rows).sort("instrument_id") if latest_rows else pl.DataFrame()
    bars_recent = (
        pl.DataFrame(recent_rows).sort(["instrument_id", "bar_start"])
        if recent_rows
        else pl.DataFrame()
    )
    feature_latest_rows = [d[-1] for d in state.published_features.values() if d]
    features_latest = (
        pl.DataFrame(feature_latest_rows).sort("instrument_id") if feature_latest_rows else None
    )
    feature_recent_rows = [r for d in state.published_features.values() for r in d]
    features_recent = (
        pl.DataFrame(feature_recent_rows).sort(["instrument_id", "bar_start"])
        if feature_recent_rows
        else None
    )
    portfolio = state.portfolio
    return DecisionContext(
        decision_at=t,
        instruments=tuple(sorted(i for i, d in state.published_bars.items() if d)),
        bars_latest=bars_latest,
        bars_recent=bars_recent,
        features_latest=features_latest,
        features_recent=features_recent,
        positions={i: p.quantity for i, p in portfolio.positions.items() if not p.is_flat},
        cash=portfolio.cash,
        equity=portfolio.equity,
        marks={i: m.price for i, m in portfolio.marks.items()},
        pending_orders=tuple(live.order for live in state.pending),
        run_id=run_id,
        fold=fold,
    )


def _check_staleness(t: _dt.datetime, state: _State, config: SimulationConfig) -> None:
    limit = config.max_mark_staleness
    if limit is None:
        return
    for instrument_id, position in state.portfolio.positions.items():
        if position.is_flat:
            continue
        mark = state.portfolio.marks.get(instrument_id)
        if mark is not None and t - mark.at > limit:
            _warn(
                state,
                t,
                "stale_mark",
                f"position valued at a mark {t - mark.at} old (limit {limit})",
                instrument_id,
            )


def _record_snapshot(t: _dt.datetime, state: _State, config: SimulationConfig) -> None:
    state.portfolio.reconcile()
    state.snapshots.append(state.portfolio.snapshot(t, pending_orders=len(state.pending)))
    state.positions.extend(state.portfolio.position_snapshots(t))


def _assert_chain(fill: Fill, state: _State) -> None:
    order = next(o for o in state.orders if o.order_id == fill.order_id)
    if not (order.signal_at <= order.order_at <= order.eligible_at <= fill.fill_at):
        raise AssertionError(
            f"timestamp chain violated for {order.order_id}: {order.signal_at} <= "
            f"{order.order_at} <= {order.eligible_at} <= {fill.fill_at}"
        )


def _order_event(
    state: _State,
    order_id: OrderId,
    at: _dt.datetime,
    status: OrderStatus,
    quantity: float,
    reason: str | None = None,
) -> None:
    state.order_events.append(
        OrderEvent(order_id=order_id, at=at, status=status, quantity=quantity, reason=reason)
    )


def _warn(
    state: _State,
    at: _dt.datetime,
    code: str,
    message: str,
    instrument_id: InstrumentId | None = None,
) -> None:
    key = (code, instrument_id)
    existing = state.warnings.get(key)
    if existing is None:
        state.warnings[key] = WarningRecord(
            at=at, code=code, message=message, instrument_id=instrument_id
        )
    else:
        state.warnings[key] = existing.model_copy(update={"occurrences": existing.occurrences + 1})
