"""HTML for the local UI.

Server-rendered with the same inline-SVG helpers the static report uses, so there is no
build step, no node toolchain, and no JavaScript framework to keep current in a Python
project. Interactivity that genuinely needs the client (filtering a table, picking runs to
overlay) is a few dozen lines of vanilla JS.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

import polars as pl

from qresearch.artifacts.charts import Series, line_chart
from qresearch.artifacts.styles import BASE_CSS

_UI_CSS = """
nav{display:flex;gap:18px;align-items:baseline;border-bottom:1px solid var(--line);
padding:10px 20px;background:var(--card);position:sticky;top:0;z-index:5}
nav a{color:var(--muted);text-decoration:none;font-size:12px;text-transform:uppercase;
letter-spacing:.06em}
nav a.on,nav a:hover{color:var(--fg)}
nav .brand{font-weight:700;letter-spacing:-.01em;text-transform:none;font-size:14px;color:var(--fg)}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
input[type=search],select{background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:5px 8px;font:12px ui-monospace,monospace}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:6px 12px;
font-size:12px;cursor:pointer}
button.ghost{background:transparent;color:var(--muted);border:1px solid var(--line)}
button:disabled{opacity:.45;cursor:not-allowed}
tbody tr{cursor:pointer}
tbody tr.sel{background:color-mix(in srgb,var(--accent) 12%,transparent)}
a.run{color:var(--accent);text-decoration:none;font-family:ui-monospace,monospace}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;border:1px solid var(--line);
font-size:11px;color:var(--muted)}
.pill.warn{color:var(--warn-fg);border-color:var(--warn-line);background:var(--warn-bg)}
.count{color:var(--muted);font-size:12px}
"""


_HAY_COLUMNS = ("run_id", "config_id", "label", "strategy", "cost_scenario", "fill_rule")
"""Columns concatenated into each row's search haystack for client-side filtering."""


def _options(values: Sequence[str]) -> str:
    return "".join(f'<option value="{_e(v)}">{_e(v)}</option>' for v in values)


def _haystack(row: dict[str, Any]) -> str:
    """Everything a row can be filtered on, concatenated for the client-side search."""
    return " ".join(str(row[column] or "") for column in _HAY_COLUMNS)


def _e(value: Any) -> str:
    return html.escape(str(value))


def _fmt(value: Any, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def _signed(value: float | None, spec: str = "+.2%") -> str:
    if value is None:
        return "—"
    css = "pos" if value > 0 else ("neg" if value < 0 else "")
    return f'<span class="{css}">{format(value, spec)}</span>'


def shell(title: str, body: str, *, active: str = "") -> str:
    """The app frame: nav, shared styles, page body."""
    links = [("runs", "/"), ("jobs", "/jobs"), ("compare", "/compare")]
    nav = "".join(
        f'<a href="{href}" class="{"on" if name == active else ""}">{name}</a>'
        for name, href in links
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{BASE_CSS}{_UI_CSS}</style></head><body>"
        f'<nav><span class="brand">qresearch</span>{nav}</nav>'
        f'<div class="wrap">{body}</div></body></html>'
    )


def runs_page(table: pl.DataFrame) -> str:
    """Run list: filter, select two or more, compare."""
    if table.is_empty():
        return shell(
            "runs",
            '<h1>No runs yet</h1><p class="sub">Run a backtest, then reload:</p>'
            '<div class="card"><code>qresearch backtest run -c configs/&lt;your&gt;.yaml '
            "--root data --runs runs</code></div>",
            active="runs",
        )

    scenarios = sorted({r for r in table.get_column("cost_scenario").to_list() if r})
    strategies = sorted({r for r in table.get_column("strategy").to_list() if r})

    rows = []
    for row in table.sort("started_at", descending=True).iter_rows(named=True):
        warn = row["warning_count"] or 0
        pill = f'<span class="pill{" warn" if warn else ""}">{warn}</span>'
        rows.append(
            f'<tr data-run="{_e(row["run_id"])}" '
            f'data-hay="{_e(_haystack(row))}">'
            f'<td><input type="checkbox" class="pick" value="{_e(row["run_id"])}"></td>'
            f'<td><a class="run" href="/runs/{_e(row["run_id"])}">{_e(row["run_id"][:16])}</a></td>'
            f"<td>{_e(row['label'] or '—')}</td>"
            f"<td>{_e(row['strategy'])}</td>"
            f"<td>{_e(row['cost_scenario'])}</td>"
            f'<td><span class="pill">{_e(row["fill_rule"])}</span></td>'
            f"<td>{row['fold_count']}</td><td>{pill}</td>"
            f"<td>{_signed(row['test_total_return'])}</td>"
            f"<td>{_fmt(row['test_sharpe'], '.2f')}</td>"
            f'<td class="neg">{_fmt(row["test_max_drawdown"], ".2%")}</td>'
            f"<td>{_fmt(row['test_cost_fraction'], '.3%')}</td>"
            "</tr>"
        )

    body = (
        f'<h1>Runs <span class="count">({table.height})</span></h1>'
        '<div class="toolbar">'
        '<input type="search" id="q" placeholder="filter by id, label, strategy…" size="34">'
        '<select id="scenario"><option value="">any scenario</option>'
        f"{_options(scenarios)}</select>"
        '<select id="strategy"><option value="">any strategy</option>'
        f"{_options(strategies)}</select>"
        '<button id="cmp" disabled>Compare selected</button>'
        '<button class="ghost" id="clear">Clear</button>'
        '<span class="count" id="shown"></span></div>'
        '<div class="card"><table><thead><tr><th></th><th>run</th><th>label</th><th>strategy</th>'
        "<th>scenario</th><th>fill rule</th><th>folds</th><th>warn</th><th>test return</th>"
        "<th>sharpe</th><th>max dd</th><th>costs</th></tr></thead>"
        f'<tbody id="rows">{"".join(rows)}</tbody></table></div>'
        f"<script>{_RUNS_JS}</script>"
    )
    return shell("runs", body, active="runs")


_RUNS_JS = """
const q=document.getElementById('q'),sc=document.getElementById('scenario'),
st=document.getElementById('strategy'),cmp=document.getElementById('cmp'),
clear=document.getElementById('clear'),shown=document.getElementById('shown');
const rows=[...document.querySelectorAll('#rows tr')];
function apply(){
  const t=q.value.toLowerCase(), s=sc.value, g=st.value; let n=0;
  for(const r of rows){
    const hay=r.dataset.hay.toLowerCase();
    const ok=(!t||hay.includes(t))&&(!s||hay.includes(s))&&(!g||hay.includes(g));
    r.style.display=ok?'':'none'; if(ok)n++;
  }
  shown.textContent=n+' shown';
}
function picked(){return [...document.querySelectorAll('.pick:checked')].map(c=>c.value);}
function sync(){cmp.disabled=picked().length<2;
  for(const r of rows) r.classList.toggle('sel', r.querySelector('.pick').checked);}
q.oninput=apply; sc.onchange=apply; st.onchange=apply;
document.getElementById('rows').addEventListener('change',sync);
document.getElementById('rows').addEventListener('click',e=>{
  if(e.target.tagName==='A'||e.target.classList.contains('pick'))return;
  const box=e.target.closest('tr').querySelector('.pick'); box.checked=!box.checked; sync();});
cmp.onclick=()=>location='/compare?runs='+picked().join(',');
clear.onclick=()=>{q.value='';sc.value='';st.value='';
  document.querySelectorAll('.pick').forEach(c=>c.checked=false);apply();sync();};
apply(); sync();
"""


def compare_page(table: pl.DataFrame, curves: dict[str, pl.DataFrame], role: str) -> str:
    """Overlaid equity curves plus a metric-by-run table."""
    if table.is_empty():
        return shell(
            "compare",
            "<h1>Nothing selected</h1>"
            '<p class="sub">Pick two or more runs on the <a href="/">runs</a> page.</p>',
            active="compare",
        )

    palette = ["#3b7dd8", "#2f9e68", "#d9822b", "#8e6fd8", "#c0392b", "#0d9488"]
    series: list[Series] = []
    for index, (run_id, curve) in enumerate(curves.items()):
        segment = curve.filter(pl.col("role") == role) if "role" in curve.columns else curve
        segment = segment.sort("at")
        if segment.is_empty():
            continue
        base = float(segment.get_column("equity")[0]) or 1.0
        series.append(
            Series(
                run_id[:16],
                [
                    (t.timestamp(), v / base - 1.0)
                    for t, v in zip(segment["at"], segment["equity"], strict=True)
                ],
                palette[index % len(palette)],
            )
        )

    metrics = [
        ("return", "test_total_return", "+.2%"),
        ("ann. return", "test_annualized_return", "+.2%"),
        ("sharpe", "test_sharpe", ".2f"),
        ("max dd", "test_max_drawdown", ".2%"),
        ("turnover", "test_turnover_annualized", ",.0f"),
        ("costs/equity", "test_cost_fraction", ".3%"),
        ("fills", "test_fill_count", ",.0f"),
        ("warnings", "warning_count", ",.0f"),
    ]
    header = "".join(f"<th>{_e(r['run_id'][:16])}</th>" for r in table.iter_rows(named=True))
    body_rows = []
    for label, column, spec in metrics:
        cells = "".join(
            "<td>"
            + (_signed(r[column], spec) if spec.startswith("+") else _fmt(r[column], spec))
            + "</td>"
            for r in table.iter_rows(named=True)
        )
        body_rows.append(f"<tr><td>{_e(label)}</td>{cells}</tr>")
    assumptions = []
    for label, column in (
        ("strategy", "strategy"),
        ("scenario", "cost_scenario"),
        ("fill rule", "fill_rule"),
        ("dataset", "dataset_id"),
        ("config", "config_id"),
    ):
        cells = "".join(
            f'<td class="mono">{_e(r[column])}</td>' for r in table.iter_rows(named=True)
        )
        assumptions.append(f"<tr><td>{_e(label)}</td>{cells}</tr>")

    other = "validation" if role == "test" else "test"
    ids = ",".join(table.get_column("run_id").to_list())
    body = (
        f'<h1>Compare <span class="count">({table.height} runs, {_e(role)})</span></h1>'
        f'<div class="toolbar"><a href="/compare?runs={_e(ids)}&role={_e(other)}">'
        f'<button class="ghost">show {_e(other)}</button></a>'
        '<a href="/"><button class="ghost">back to runs</button></a></div>'
        f'<div class="card">{line_chart(series, y_format="percent", zero_line=True)}'
        '<div class="legend">equity as return from each segment start</div></div>'
        '<h2>Assumptions</h2><div class="card"><table><thead><tr><th></th>'
        f"{header}</tr></thead><tbody>{''.join(assumptions)}</tbody></table></div>"
        '<h2>Performance</h2><div class="card"><table><thead><tr><th></th>'
        f"{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )
    return shell("compare", body, active="compare")


def not_found(message: str) -> str:
    return shell("not found", f'<h1>Not found</h1><p class="sub">{_e(message)}</p>')


__all__ = [
    "Sequence",
    "compare_page",
    "job_log_page",
    "jobs_page",
    "not_found",
    "runs_page",
    "shell",
]


_STATE_COLOUR = {
    "queued": "",
    "running": "",
    "succeeded": "pos",
    "failed": "neg",
    "cancelled": "",
}


def jobs_page(jobs: Sequence[Any]) -> str:
    """Submitted work, newest first, with live progress and the exact command."""
    if not jobs:
        return shell(
            "jobs",
            "<h1>No jobs</h1>"
            '<p class="sub">Jobs appear here when a backtest is launched from the UI or '
            "with <code>qresearch jobs submit</code>.</p>",
            active="jobs",
        )

    rows = []
    for job in jobs:
        progress = f"{job.folds_complete} folds" if job.folds_complete else "—"
        runs = (
            " ".join(f'<a class="run" href="/runs/{_e(r)}">{_e(r[:14])}</a>' for r in job.run_ids)
            or "—"
        )
        duration = f"{job.duration.total_seconds():.1f}s" if job.duration else "—"
        cancel = (
            f'<button class="ghost cancel" data-job="{_e(job.job_id)}">cancel</button>'
            if not job.state.terminal
            else ""
        )
        rows.append(
            f'<tr><td><span class="pill">{_e(job.kind.value)}</span></td>'
            f'<td class="{_STATE_COLOUR.get(job.state.value, "")}">{_e(job.state.value)}</td>'
            f"<td>{_e(job.label or '—')}</td><td>{progress}</td><td>{runs}</td>"
            f"<td>{duration}</td><td>{_e(job.message or '—')}</td>"
            f'<td><a href="/jobs/{_e(job.job_id)}">log</a> {cancel}</td></tr>'
        )

    active = any(not j.state.terminal for j in jobs)
    body = (
        f'<h1>Jobs <span class="count">({len(jobs)})</span></h1>'
        + ('<p class="sub">Refreshing while work is in flight.</p>' if active else "")
        + '<div class="card"><table><thead><tr><th>kind</th><th>state</th><th>label</th>'
        "<th>progress</th><th>runs</th><th>took</th><th>latest</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f"<script>{_JOBS_JS}</script>"
    )
    return shell("jobs", body, active="jobs")


_JOBS_JS = """
document.querySelectorAll('.cancel').forEach(b=>b.onclick=async e=>{
  e.preventDefault(); b.disabled=true;
  await fetch('/api/jobs/'+b.dataset.job,{method:'DELETE'}); location.reload();});
if(document.querySelector('.cancel')) setTimeout(()=>location.reload(), 2000);
"""


def job_log_page(job: Any, lines: Sequence[dict[str, Any]], output: str) -> str:
    """One job: what it ran, and what it said."""

    def _row(line: dict[str, Any]) -> str:
        noise = {"time", "message", "level", "logger"}
        fields = {k: v for k, v in line.items() if k not in noise}
        return (
            f"<tr><td>{_e(line.get('time', ''))}</td>"
            f"<td>{_e(line.get('message', ''))}</td>"
            f'<td class="mono">{_e(fields)}</td></tr>'
        )

    events = "".join(_row(line) for line in lines)
    command = " ".join(job.command)
    body = (
        f"<h1>{_e(job.label or job.kind.value)} "
        f'<span class="count">{_e(job.state.value)}</span></h1>'
        f'<div class="sub mono">{_e(job.job_id)}</div>'
        '<h2>Command</h2><div class="card"><code>' + _e(command) + "</code>"
        '<div class="legend">run this yourself and you get the same result</div></div>'
        + (
            f'<h2>Events</h2><div class="card"><table><thead><tr><th>time</th><th>message</th>'
            f"<th>fields</th></tr></thead><tbody>{events}</tbody></table></div>"
            if events
            else ""
        )
        + f'<h2>Output</h2><div class="card"><pre class="mono">{_e(output[-20000:])}</pre></div>'
        '<div class="toolbar"><a href="/jobs"><button class="ghost">back to jobs</button></a></div>'
    )
    return shell("job", body, active="jobs")
