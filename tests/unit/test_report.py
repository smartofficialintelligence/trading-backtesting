"""Report rendering: structure, ordering, escaping, and downsampling."""

from __future__ import annotations

import datetime as dt
from html.parser import HTMLParser
from typing import ClassVar

import polars as pl
import pytest

from qresearch.artifacts.charts import MAX_POINTS, downsample, histogram, line_chart
from qresearch.artifacts.contracts import RunResult, RunSpec, RunStatus, StrategyRef
from qresearch.artifacts.report import render_report
from qresearch.ids import DatasetId
from qresearch.research.metrics import AnnualizationPolicy, Metrics
from qresearch.research.splits import TimeRange
from qresearch.research.walk_forward import WalkForwardPlan
from qresearch.simulation.engine import SimulationConfig
from qresearch.simulation.events import WarningRecord

T0 = dt.datetime(2024, 3, 4, tzinfo=dt.UTC)
TS = pl.Datetime("us", "UTC")


class _Balanced(HTMLParser):
    VOID: ClassVar[frozenset[str]] = frozenset(
        {"meta", "br", "img", "line", "rect", "path", "input", "hr"}
    )

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.mismatched: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            self.mismatched.append(tag)
            self.stack.remove(tag)


def spec(**overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "dataset_id": DatasetId("ds_demo"),
        "instrument_ids": ("X",),
        "bar_size": "1m",
        "calendar_id": "24x7:1",
        "strategy": StrategyRef(kind="buy_and_hold"),
        "simulation": SimulationConfig(),
        "plan": WalkForwardPlan(
            train=dt.timedelta(hours=2), test=dt.timedelta(hours=1), purge=dt.timedelta(minutes=1)
        ),
        "span": TimeRange(start=T0, end=T0 + dt.timedelta(hours=6)),
        "label": "demo run",
    }
    return RunSpec.model_validate(base | overrides)


def metrics(**overrides: object) -> Metrics:
    base: dict[str, object] = {
        "periods": 100,
        "start_equity": 100.0,
        "end_equity": 110.0,
        "total_return": 0.1,
        "annualized_return": 0.5,
        "annualized_volatility": 0.2,
        "sharpe": 1.4,
        "max_drawdown": 0.05,
        "max_drawdown_duration": dt.timedelta(minutes=30),
        "turnover_annualized": 12.0,
        "avg_gross_exposure": 0.5,
        "avg_net_exposure": 0.5,
        "order_count": 10,
        "fill_count": 10,
        "period_hit_rate": 0.55,
        "total_costs": 1.0,
        "cost_fraction": 0.01,
        "mean_participation": 0.02,
        "warning_count": 1,
        "annualization": AnnualizationPolicy(
            periods_per_year=525960.0, label="24x7 calendar, 1m bars"
        ),
    }
    return Metrics.model_validate(base | overrides)


def result(**overrides: object) -> RunResult:
    base: dict[str, object] = {
        "run_id": spec().run_id,
        "status": RunStatus.COMPLETE,
        "started_at": T0,
        "finished_at": T0 + dt.timedelta(minutes=1),
        "aggregate": {"test": metrics(), "validation": metrics(total_return=-0.02)},
        "economic_digest": "a" * 64,
    }
    return RunResult.model_validate(base | overrides)


def frames(points: int = 200) -> dict[str, pl.DataFrame]:
    stamps = [T0 + dt.timedelta(minutes=i) for i in range(points)]
    curve = pl.DataFrame(
        {
            "at": stamps * 2,
            "equity": [100.0 + i * 0.05 for i in range(points)]
            + [100.0 - i * 0.01 for i in range(points)],
            "gross_exposure": [50.0] * points * 2,
            "net_exposure": [50.0] * points * 2,
            "cash": [50.0] * points * 2,
            "role": ["test"] * points + ["validation"] * points,
            "fold": [0] * points * 2,
        },
        schema_overrides={"at": TS},
    )
    fills = pl.DataFrame(
        {
            "fill_id": ["f1", "f2"],
            "instrument_id": ["X", "X"],
            "side": ["buy", "sell"],
            "quantity": [1.0, 1.0],
            "fill_at": [T0, T0 + dt.timedelta(minutes=5)],
            "reference_price": [100.0, 110.0],
            "half_spread": [0.01, 0.01],
            "slippage": [0.02, 0.02],
            "price": [100.03, 109.97],
            "fee": [0.1, 0.1],
            "role": ["test", "test"],
            "fold": [0, 0],
        },
        schema_overrides={"fill_at": TS},
    )
    trades = pl.DataFrame(
        {
            "trade_id": [1],
            "instrument_id": ["X"],
            "direction": ["long"],
            "opened_at": [T0],
            "closed_at": [T0 + dt.timedelta(minutes=5)],
            "duration_s": [300.0],
            "peak_quantity": [1.0],
            "entry_price": [100.03],
            "exit_price": [109.97],
            "fill_count": [2],
            "gross_pnl": [10.0],
            "costs": [0.26],
            "net_pnl": [9.74],
            "return_on_notional": [0.0974],
            "is_open": [False],
            "fold": [0],
            "role": ["test"],
        },
        schema_overrides={"opened_at": TS, "closed_at": TS},
    )
    folds = pl.DataFrame(
        {
            "fold": [0, 0],
            "role": ["train", "test"],
            "start": [T0, T0 + dt.timedelta(hours=2)],
            "end": [T0 + dt.timedelta(hours=2), T0 + dt.timedelta(hours=3)],
        },
        schema_overrides={"start": TS, "end": TS},
    )
    return {"equity_curve": curve, "fills": fills, "trades": trades, "folds": folds}


# -- structure ---------------------------------------------------------------------------


def test_the_report_is_well_formed_and_self_contained() -> None:
    text = render_report(spec(), result(), frames())
    parser = _Balanced()
    parser.feed(text)
    assert parser.stack == [] and parser.mismatched == []
    assert text.startswith("<!doctype html>")
    for remote in ("http://", "https://", "//cdn", "<script"):
        assert remote not in text, f"report must be self-contained, found {remote!r}"


def test_assumptions_appear_before_performance() -> None:
    """The layout is the cheapest place to stop a Sharpe being read without its caveats."""
    text = render_report(spec(), result(), frames())
    assumptions = text.index("What this run assumed")
    warnings_at = text.index("fill rule")
    headline = text.index("headline")
    assert assumptions < warnings_at < headline


def test_warnings_are_rendered_above_the_metrics() -> None:
    warned = result(
        warnings=(
            WarningRecord(at=T0, code="static_universe", message="survivorship", occurrences=1),
            WarningRecord(at=T0, code="optimistic_fill_rule", message="optimistic", occurrences=8),
        )
    )
    text = render_report(spec(), warned, frames())
    assert "9 warning(s)" in text
    assert "static_universe" in text and "optimistic_fill_rule" in text
    assert text.index("warning(s)") < text.index("headline")


def test_key_sections_and_charts_are_present() -> None:
    text = render_report(spec(), result(), frames())
    for section in ("Equity", "Drawdown and exposure", "Where the money went", "Largest trades"):
        assert section in text, section
    assert text.count("<svg") >= 5


def test_a_run_without_fills_or_trades_still_renders() -> None:
    minimal = {"equity_curve": frames()["equity_curve"]}
    text = render_report(spec(), result(), minimal)
    parser = _Balanced()
    parser.feed(text)
    assert parser.stack == []
    assert "no fills" in text or "no closed trades" in text


def test_empty_frames_render_without_crashing() -> None:
    text = render_report(spec(), result(aggregate={}), {})
    parser = _Balanced()
    parser.feed(text)
    assert parser.stack == []


# -- escaping ----------------------------------------------------------------------------


def test_untrusted_text_is_escaped() -> None:
    """A label or warning message ends up in HTML; it must not be able to inject markup."""
    nasty = '<img src=x onerror="alert(1)">'
    text = render_report(
        spec(label=nasty),
        result(warnings=(WarningRecord(at=T0, code="c", message=nasty, occurrences=1),)),
        frames(),
    )
    # The property that matters is that no *tag* can be created: '<' is escaped, so the
    # payload survives only as inert text. Asserting the absence of the substring
    # "onerror=" would be theatre -- it appears escaped and harmless.
    assert "<img" not in text, "a raw tag reached the document"
    assert text.count("&lt;img") >= 2, "escaped in both the title and the warning list"
    parser = _Balanced()
    parser.feed(text)
    assert parser.stack == [] and parser.mismatched == [], "injection would unbalance the tree"


# -- downsampling --------------------------------------------------------------------------


def test_downsample_keeps_endpoints_and_extremes() -> None:
    points = [(float(i), float(i % 7)) for i in range(20_000)]
    points[5000] = (5000.0, 999.0)  # a spike that must survive
    points[9000] = (9000.0, -999.0)
    reduced = downsample(points)
    assert len(reduced) <= MAX_POINTS + 2
    assert reduced[0] == points[0] and reduced[-1] == points[-1]
    values = [y for _, y in reduced]
    assert 999.0 in values and -999.0 in values, "stride sampling would lose these"
    assert [x for x, _ in reduced] == sorted(x for x, _ in reduced), "x must stay ordered"


def test_downsample_is_a_no_op_below_the_limit() -> None:
    points = [(float(i), float(i)) for i in range(50)]
    assert downsample(points) == points


def test_charts_handle_degenerate_input() -> None:
    from qresearch.artifacts.charts import Series

    assert "no data" in line_chart([]) or "empty" in line_chart([])
    assert "<svg" in line_chart([Series("flat", [(0.0, 1.0), (1.0, 1.0)], "#000")])
    assert "empty" in histogram([])


@pytest.mark.parametrize("points", [1, 2, 3])
def test_tiny_series_render(points: int) -> None:
    text = render_report(spec(), result(), frames(points=points))
    parser = _Balanced()
    parser.feed(text)
    assert parser.stack == []
