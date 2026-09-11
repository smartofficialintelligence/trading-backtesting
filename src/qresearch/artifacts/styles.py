"""Shared stylesheet for the static report and the local UI.

One source so a run looks the same whether it is read from an archived ``report.html`` or
through the browser app. Light and dark are both defined here; the viewer's system
setting decides, since a research tool gets read at all hours.
"""

from __future__ import annotations

from typing import Final

BASE_CSS: Final = """
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
