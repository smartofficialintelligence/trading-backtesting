"""Self-contained HTML report for one run.

Written into the run directory as ``report.html``, so it is versioned with the run and
reproducible from it — the same append-only artifact model as every other file there. No
server, no CDN, no plotting dependency: inline CSS and inline SVG only, so the file still
opens in ten years.

Ordering is deliberate and mirrors ``docs/leakage_checklist.md``: **what this run assumed
comes before what it earned.** The header states the dataset, cost scenario, fill rule and
code revision; warnings sit above the metrics, not in a footnote. A headline Sharpe read
without its assumptions is how people fool themselves, and the layout is the cheapest
place to make that hard.
"""

from __future__ import annotations

import datetime as _dt
import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from qresearch.artifacts.charts import Series, bar_chart, fold_ribbon, histogram, line_chart
from qresearch.artifacts.contracts import RunResult, RunSpec
from qresearch.time import now_utc

ROLE_COLOURS = {"validation": "#8e6fd8", "test": "#2f9e68"}
COST_COLOURS = ("#d9822b", "#c0392b", "#8e6fd8")

_CSS = """
:root{--bg:#fbfbfd;--fg:#16181d;--muted:#5d6470;--line:#e2e5ea;--card:#fff;
--warn-bg:#fff6e6;--warn-fg:#8a5a00;--warn-line:#f0c987;--good:#2f9e68;--bad:#c0392b;
--accent:#3b7dd8}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8ec;--muted:#98a1af;--line:#2a2f38;
--card:#1b1e24;--warn-bg:#2e2413;--warn-fg:#e3b667;--warn-line:#5c4820}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:17px;margin:0 0 2px;letter-spacing:-.01em}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
margin:28px 0 10px;font-weight:600}
.sub{color:var(--muted);font-size:12px;margin-bottom:14px}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;
margin-bottom:14px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:6px 20px}
.fact{display:flex;justify-content:space-between;gap:12px;padding:3px 0;
border-bottom:1px solid var(--line)}
.fact span:first-child{color:var(--muted)}
.fact span:last-child{font-family:ui-monospace,monospace;text-align:right}
.warn{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-fg);
border-radius:8px;padding:12px 16px;margin-bottom:14px}
.warn b{display:block;margin-bottom:6px}
.warn li{margin:3px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-family:ui-monospace,monospace;font-size:16px;margin-top:3px}
.pos{color:var(--good)}.neg{color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
tbody tr:hover{background:var(--bg)}
.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.tick{fill:var(--muted);font-size:10px;font-family:ui-monospace,monospace}
.barval{fill:var(--fg);font-size:10px;font-family:ui-monospace,monospace}
.legend{margin-top:6px;color:var(--muted);font-size:11px}
.key{margin-right:14px}.key i{display:inline-block;width:9px;height:9px;border-radius:2px;
margin-right:5px}
.empty{color:var(--muted);font-style:italic;padding:18px 0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:720px){.two{grid-template-columns:1fr}}
footer{color:var(--muted);font-size:11px;margin-top:34px;border-top:1px solid var(--line);
padding-top:12px}
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _signed(value: float | None, spec: str = "+.2%") -> str:
    if value is None:
        return '<span class="muted">n/a</span>'
    css = "pos" if value > 0 else ("neg" if value < 0 else "")
    return f'<span class="{css}">{format(value, spec)}</span>'


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _fact(label: str, value: str) -> str:
    return f'<div class="fact"><span>{_e(label)}</span><span>{value}</span></div>'


def _tile(key: str, value: str) -> str:
    return f'<div class="tile"><div class="k">{_e(key)}</div><div class="v">{value}</div></div>'


def _x_labels(frame: pl.DataFrame, column: str, count: int = 5) -> list[tuple[float, str]]:
    if frame.is_empty():
        return []
    stamps = frame.get_column(column).to_list()
    picks = (
        [stamps[round(i * (len(stamps) - 1) / (count - 1))] for i in range(count)]
        if len(stamps) > 1
        else stamps
    )
    span = (stamps[-1] - stamps[0]).total_seconds() if len(stamps) > 1 else 0
    fmt = "%H:%M" if span < 3 * 86400 else "%m-%d"
    return [(s.timestamp(), s.strftime(fmt)) for s in picks]


def _curve_for(curve: pl.DataFrame, role: str) -> pl.DataFrame:
    return (
        curve.filter(pl.col("role") == role).sort("at")
        if "role" in curve.columns
        else curve.sort("at")
    )


def _normalised(frame: pl.DataFrame) -> list[tuple[float, float]]:
    """Equity as a return from the segment's start, so folds are comparable."""
    if frame.is_empty():
        return []
    base = float(frame.get_column("equity")[0]) or 1.0
    return [
        (t.timestamp(), v / base - 1.0) for t, v in zip(frame["at"], frame["equity"], strict=True)
    ]


def _drawdown(frame: pl.DataFrame) -> list[tuple[float, float]]:
    points, peak = [], None
    for stamp, value in zip(frame["at"], frame["equity"], strict=True):
        peak = value if peak is None else max(peak, value)
        points.append((stamp.timestamp(), value / peak - 1.0))
    return points


def render_report(
    spec: RunSpec,
    result: RunResult,
    frames: Mapping[str, pl.DataFrame],
    *,
    environment: Any = None,
) -> str:
    """Build the report HTML. Pure: takes loaded artifacts, returns a string."""
    curve = frames.get("equity_curve", pl.DataFrame())
    trades = frames.get("trades", pl.DataFrame())
    fills = frames.get("fills", pl.DataFrame())
    folds = frames.get("folds", pl.DataFrame())
    roles = [r for r in ("test", "validation") if r in result.aggregate]

    body: list[str] = ['<div class="wrap">']
    body.append(
        f"<h1>{_e(spec.label or spec.strategy.kind)}</h1>"
        f'<div class="sub mono">{_e(result.run_id)} &middot; {_e(spec.config_id)} &middot; '
        f"{_e(result.status.value)}</div>"
    )

    # -- assumptions first -------------------------------------------------------------
    execution = spec.simulation.execution
    costs = execution.costs
    body.append('<h2>What this run assumed</h2><div class="card"><div class="facts">')
    body.append(_fact("dataset", f"<code>{_e(spec.dataset_id)}</code>"))
    body.append(_fact("instruments", _e(", ".join(spec.instrument_ids))))
    body.append(_fact("bars / calendar", f"{_e(spec.bar_size)} &middot; {_e(spec.calendar_id)}"))
    body.append(_fact("strategy", f"{_e(spec.strategy.kind)} {_e(spec.strategy.params)}"))
    body.append(_fact("features", _e([f.kind for f in spec.features]) if spec.features else "none"))
    body.append(
        _fact("transforms", _e([t.kind for t in spec.transforms]) if spec.transforms else "none")
    )
    body.append(_fact("cost scenario", _e(spec.cost_scenario)))
    body.append(_fact("fill rule", f"<b>{_e(execution.fill_rule.value)}</b>"))
    body.append(_fact("latency", f"{execution.submission_latency} + {execution.order_latency}"))
    body.append(
        _fact(
            "spread / commission", f"{costs.half_spread_bps:g} bps / {costs.commission_bps:g} bps"
        )
    )
    body.append(
        _fact("slippage", f"{costs.slippage.kind.value} &times;{costs.slippage.coefficient:g}")
    )
    body.append(_fact("participation cap", _e(execution.participation_cap or "none")))
    body.append(_fact("code revision", f"<code>{_e(spec.code_revision or '-')}</code>"))
    body.append("</div></div>")

    if result.warnings:
        items = "".join(
            f"<li><code>{_e(w.code)}</code>"
            + (f" [{_e(w.instrument_id)}]" if w.instrument_id else "")
            + f" &times;{w.occurrences} — {_e(w.message)}</li>"
            for w in result.warnings
        )
        total = sum(w.occurrences for w in result.warnings)
        body.append(
            f'<div class="warn"><b>{total} warning(s) — read before the numbers</b>'
            f"<ul>{items}</ul></div>"
        )

    # -- headline ----------------------------------------------------------------------
    for role in roles:
        metrics = result.aggregate[role]
        body.append(f"<h2>{_e(role)} — headline</h2>")
        tiles = [
            _tile("return", _signed(metrics.total_return)),
            _tile("ann. return", _signed(metrics.annualized_return)),
            _tile("ann. vol", _fmt(metrics.annualized_volatility, ".1%")),
            _tile("sharpe", _fmt(metrics.sharpe, ".2f")),
            _tile("max drawdown", f'<span class="neg">{metrics.max_drawdown:.2%}</span>'),
            _tile("costs / equity", f'<span class="neg">{metrics.cost_fraction:.3%}</span>'),
            _tile("turnover p.a.", _fmt(metrics.turnover_annualized, ",.0f")),
            _tile("fills", f"{metrics.fill_count:,}"),
        ]
        if metrics.trades is not None:
            stats = metrics.trades
            tiles += [
                _tile("trades", f"{stats.trade_count:,}"),
                _tile("win rate", _fmt(stats.win_rate, ".1%")),
                _tile("profit factor", _fmt(stats.profit_factor, ".2f")),
                _tile("median hold", f"{(stats.median_duration_s or 0) / 60:.0f}m"),
            ]
        body.append(f'<div class="tiles">{"".join(tiles)}</div>')

    # -- curves ------------------------------------------------------------------------
    if not curve.is_empty():
        body.append("<h2>Equity</h2>")
        series = [
            Series(
                role,
                _normalised(_curve_for(curve, role)),
                ROLE_COLOURS.get(role, "#3b7dd8"),
                fill=True,
            )
            for role in roles
        ] or [Series("equity", _normalised(curve.sort("at")), "#3b7dd8", fill=True)]
        body.append(
            '<div class="card">'
            + line_chart(
                series,
                y_format="percent",
                x_labels=_x_labels(curve.sort("at"), "at"),
                zero_line=True,
                title="equity, as return from segment start",
            )
            + "</div>"
        )
        body.append("<h2>Drawdown and exposure</h2>")
        dd = [
            Series(
                role,
                _drawdown(_curve_for(curve, role)),
                ROLE_COLOURS.get(role, "#c0392b"),
                fill=True,
            )
            for role in roles
        ] or [Series("drawdown", _drawdown(curve.sort("at")), "#c0392b", fill=True)]
        exposure = []
        segments: list[tuple[str, pl.DataFrame]] = (
            [(role, _curve_for(curve, role)) for role in roles]
            if roles
            else [("gross", curve.sort("at"))]
        )
        for segment_role, segment in segments:
            if segment.is_empty():
                continue
            exposure.append(
                Series(
                    f"{segment_role} gross",
                    [
                        (t.timestamp(), g / e if e else 0.0)
                        for t, g, e in zip(
                            segment["at"], segment["gross_exposure"], segment["equity"], strict=True
                        )
                    ],
                    ROLE_COLOURS.get(role, "#3b7dd8"),
                )
            )
        body.append(
            '<div class="two"><div class="card">'
            + line_chart(
                dd, height=180, y_format="percent", x_labels=_x_labels(curve.sort("at"), "at")
            )
            + '<div class="legend">drawdown from running peak</div></div><div class="card">'
            + line_chart(
                exposure, height=180, y_format="percent", x_labels=_x_labels(curve.sort("at"), "at")
            )
            + '<div class="legend">gross exposure / equity</div></div></div>'
        )

    # -- costs and trades ---------------------------------------------------------------
    body.append("<h2>Where the money went</h2>")
    left = '<div class="empty">no fills</div>'
    if not fills.is_empty():
        spread = float((fills["quantity"] * fills["half_spread"]).sum())
        slip = float((fills["quantity"] * fills["slippage"]).sum())
        fee = float(fills["fee"].sum())
        left = bar_chart(["spread", "slippage", "fees"], [spread, slip, fee], list(COST_COLOURS))
        left += (
            f'<div class="legend">total friction '
            f"{spread + slip + fee:,.2f} across {fills.height:,} fills</div>"
        )
    right = '<div class="empty">no closed trades</div>'
    if not trades.is_empty():
        closed = trades.filter(~pl.col("is_open"))
        if not closed.is_empty():
            right = histogram(closed.get_column("net_pnl").drop_nulls().to_list())
            right += '<div class="legend">net P&amp;L per closed trade</div>'
    body.append(
        f'<div class="two"><div class="card">{left}</div><div class="card">{right}</div></div>'
    )

    # -- per-fold stability --------------------------------------------------------------
    if result.folds:
        body.append("<h2>Stability across folds</h2>")
        rows = "".join(
            f"<tr><td>{f.fold}</td><td>{_e(f.role.value)}</td>"
            f'<td class="mono">{f.range.start:%Y-%m-%d %H:%M}</td>'
            f"<td>{f.decisions:,}</td><td>{_signed(f.metrics.total_return)}</td>"
            f"<td>{_fmt(f.metrics.sharpe, '.2f')}</td>"
            f'<td class="neg">{f.metrics.max_drawdown:.2%}</td>'
            f"<td>{f.metrics.fill_count:,}</td>"
            f"<td>{_fmt(f.metrics.trades.win_rate if f.metrics.trades else None, '.0%')}</td></tr>"
            for f in result.folds
        )
        body.append(
            '<div class="card"><table><thead><tr><th>fold</th><th>role</th><th>starts</th>'
            "<th>decisions</th><th>return</th><th>sharpe</th><th>max dd</th><th>fills</th>"
            f"<th>win rate</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    if not folds.is_empty():
        entries = [
            (int(r["fold"]), str(r["role"]), r["start"], r["end"])
            for r in folds.iter_rows(named=True)
        ]
        body.append(f'<div class="card">{fold_ribbon(entries)}</div>')

    # -- trades ---------------------------------------------------------------------------
    if not trades.is_empty():
        closed = trades.filter(~pl.col("is_open")).sort("net_pnl", descending=True)
        if not closed.is_empty():
            body.append("<h2>Largest trades</h2>")
            shown = pl.concat([closed.head(5), closed.tail(5)]).unique(
                subset=["trade_id"], keep="first"
            )
            rows = "".join(
                f'<tr><td class="mono">{_e(r["instrument_id"])}</td><td>{_e(r["direction"])}</td>'
                f'<td class="mono">{r["opened_at"]:%m-%d %H:%M}</td>'
                f"<td>{(r['duration_s'] or 0) / 60:.0f}m</td>"
                f"<td>{r['peak_quantity']:.6g}</td><td>{r['entry_price']:,.2f}</td>"
                f"<td>{(r['exit_price'] or 0):,.2f}</td>"
                f'<td class="neg">{r["costs"]:,.2f}</td>'
                f"<td>{_signed(r['net_pnl'], '+,.2f')}</td></tr>"
                for r in shown.sort("net_pnl", descending=True).iter_rows(named=True)
            )
            body.append(
                '<div class="card"><table><thead><tr><th>instrument</th>'
                "<th>side</th><th>opened</th>"
                "<th>held</th><th>size</th><th>entry</th><th>exit</th>"
                "<th>costs</th><th>net P&amp;L</th>"
                f"</tr></thead><tbody>{rows}</tbody></table></div>"
            )

    body.append(_footer(spec, result, environment))
    body.append("</div>")
    title = f"{spec.label or spec.strategy.kind} — {result.run_id}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
        + "".join(body)
        + "</body></html>"
    )


def _footer(spec: RunSpec, result: RunResult, environment: Any) -> str:
    parts = [
        f"generated {now_utc():%Y-%m-%d %H:%M} UTC",
        f"digest <code>{_e((result.economic_digest or '')[:16])}</code>",
        f"seed {spec.seed}",
    ]
    if environment is not None:
        parts.append(f"python {_e(environment.python)}")
        if environment.git_commit:
            dirty = " (dirty)" if environment.git_dirty else ""
            parts.append(f"commit <code>{_e(environment.git_commit[:12])}</code>{dirty}")
    annual = next(iter(result.aggregate.values())).annualization.label if result.aggregate else None
    if annual:
        parts.append(_e(annual))
    return (
        "<footer>"
        + " &middot; ".join(parts)
        + "<br>Assumptions above are not incidental: see docs/leakage_checklist.md before "
        "acting on any number here.</footer>"
    )


def write_report(run_directory: Path, html_text: str) -> Path:
    """Write ``report.html`` into a run directory and return its path."""
    target = Path(run_directory) / "report.html"
    target.write_text(html_text, encoding="utf-8")
    return target


def build_report_for_run(store: Any, run_id: str) -> Path:
    """Load a stored run and write its report."""
    spec, result = store.load_spec(run_id), store.load_result(run_id)
    frames: dict[str, pl.DataFrame] = {}
    for name in ("equity_curve", "fills", "trades", "folds", "positions", "orders"):
        try:
            frames[name] = store.load_frame(run_id, name)
        except (FileNotFoundError, OSError):
            continue
    try:
        environment = store.load_environment(run_id)
    except (FileNotFoundError, OSError):
        environment = None
    return write_report(
        store.path_for(run_id), render_report(spec, result, frames, environment=environment)
    )


__all__ = ["Sequence", "_dt", "build_report_for_run", "render_report", "write_report"]
