"""Minimal inline-SVG charting for run reports.

Hand-rolled rather than matplotlib/plotly on purpose: a report is an *artifact* that
lives in the run directory and must stay readable years later without a runtime, a CDN,
or a pinned plotting library. Inline SVG in a self-contained file achieves that; it also
keeps the report free of any dependency the platform does not already have.

These are deliberately small. They draw one chart each, take pre-computed numbers, and
know nothing about runs or metrics.
"""

from __future__ import annotations

import datetime as _dt
import html
from collections.abc import Sequence
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    points: Sequence[Point]
    colour: str
    fill: bool = False
    dashed: bool = False


MAX_POINTS = 1400
"""Rendered points per series. A minute-resolution year is ~500k points; drawing them all
makes a report tens of megabytes for no visible gain."""


def downsample(points: Sequence[Point], limit: int = MAX_POINTS) -> list[Point]:
    """Reduce a series for drawing while keeping its shape.

    Plain stride sampling would drop the spikes that matter most in an equity or drawdown
    series -- the one bar where a position blew out is exactly the point a reader is
    looking for. This buckets by position and keeps each bucket's minimum and maximum in
    x order, so extremes survive.
    """
    count = len(points)
    if count <= limit:
        return list(points)
    buckets = max(limit // 2, 1)
    size = count / buckets
    out: list[Point] = [points[0]]
    for i in range(buckets):
        chunk = points[int(i * size) : max(int((i + 1) * size), int(i * size) + 1)]
        if not chunk:
            continue
        low = min(chunk, key=lambda p: p[1])
        high = max(chunk, key=lambda p: p[1])
        out.extend(sorted({low, high}, key=lambda p: p[0]))
    out.append(points[-1])
    seen: set[float] = set()
    unique: list[Point] = []
    for point in out:
        if point[0] not in seen:
            seen.add(point[0])
            unique.append(point)
    return unique


def _nice_bounds(low: float, high: float) -> tuple[float, float]:
    if low == high:
        pad = abs(low) * 0.05 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.06
    return low - pad, high + pad


def _fmt_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4g}"


def line_chart(
    series: Sequence[Series],
    *,
    width: int = 860,
    height: int = 220,
    y_format: str = "number",
    x_labels: Sequence[tuple[float, str]] = (),
    zero_line: bool = False,
    title: str = "",
) -> str:
    """A multi-series line/area chart. ``points`` are (x, y) in data space."""
    series = [Series(s.label, downsample(s.points), s.colour, s.fill, s.dashed) for s in series]
    usable = [s for s in series if s.points]
    if not usable:
        return f'<div class="empty">{html.escape(title or "no data")}</div>'

    left, right, top, bottom = 62, 12, 10, 24
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [p[0] for s in usable for p in s.points]
    ys = [p[1] for s in usable for p in s.points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = _nice_bounds(min(ys), max(ys))
    if zero_line:
        y_min, y_max = min(y_min, 0.0), max(y_max, 0.0)
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / x_span * plot_w

    def sy(y: float) -> float:
        return top + (1 - (y - y_min) / y_span) * plot_h

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="{html.escape(title)}">'
    ]
    # Horizontal gridlines with value labels.
    for i in range(5):
        value = y_min + y_span * i / 4
        y = sy(value)
        label = f"{value:.2%}" if y_format == "percent" else _fmt_number(value)
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 6}" y="{y + 3.5:.1f}" text-anchor="end">{label}</text>'
        )
    if zero_line and y_min < 0 < y_max:
        y = sy(0.0)
        parts.append(
            f'<line class="zero" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
    for x_value, label in x_labels:
        x = sx(x_value)
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 6}" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )

    for item in usable:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(x):.2f},{sy(y):.2f}"
            for i, (x, y) in enumerate(item.points)
        )
        if item.fill:
            base = sy(max(y_min, 0.0))
            first, last = item.points[0], item.points[-1]
            area = f"{path} L{sx(last[0]):.2f},{base:.2f} L{sx(first[0]):.2f},{base:.2f} Z"
            parts.append(f'<path d="{area}" fill="{item.colour}" opacity="0.16"/>')
        dash = ' stroke-dasharray="4 3"' if item.dashed else ""
        parts.append(
            f'<path d="{path}" fill="none" stroke="{item.colour}" stroke-width="1.5"{dash}/>'
        )

    parts.append("</svg>")
    if len(usable) > 1:
        legend = " ".join(
            f'<span class="key"><i style="background:{s.colour}"></i>{html.escape(s.label)}</span>'
            for s in usable
        )
        parts.append(f'<div class="legend">{legend}</div>')
    return "".join(parts)


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    colours: Sequence[str],
    *,
    width: int = 420,
    height: int = 200,
    value_format: str = "number",
) -> str:
    """A simple vertical bar chart, used for cost attribution."""
    if not values or all(v == 0 for v in values):
        return '<div class="empty">nothing to attribute</div>'
    left, right, top, bottom = 62, 12, 10, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    high = max(max(values), 0.0)
    low = min(min(values), 0.0)
    span = (high - low) or 1.0
    slot = plot_w / len(values)
    bar_w = slot * 0.6

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    for i in range(4):
        value = low + span * i / 3
        y = top + (1 - (value - low) / span) * plot_h
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        text = f"{value:.2%}" if value_format == "percent" else _fmt_number(value)
        parts.append(
            f'<text class="tick" x="{left - 6}" y="{y + 3.5:.1f}" text-anchor="end">{text}</text>'
        )
    base = top + (1 - (0 - low) / span) * plot_h
    for i, (label, value, colour) in enumerate(zip(labels, values, colours, strict=True)):
        x = left + slot * i + (slot - bar_w) / 2
        y = top + (1 - (value - low) / span) * plot_h
        parts.append(
            f'<rect x="{x:.1f}" y="{min(y, base):.1f}" width="{bar_w:.1f}" '
            f'height="{abs(base - y):.1f}" fill="{colour}" opacity="0.85"/>'
        )
        parts.append(
            f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{height - 18}" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )
        parts.append(
            f'<text class="barval" x="{x + bar_w / 2:.1f}" y="{height - 6}" '
            f'text-anchor="middle">{_fmt_number(value)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def histogram(
    values: Sequence[float],
    *,
    bins: int = 31,
    width: int = 420,
    height: int = 200,
    positive: str = "#2f9e68",
    negative: str = "#c0392b",
) -> str:
    """Distribution of trade P&L, split at zero so wins and losses read at a glance."""
    data = [v for v in values if v is not None]
    if not data:
        return '<div class="empty">no closed trades</div>'
    low, high = min(data), max(data)
    if low == high:
        low, high = low - 1, high + 1
    step = (high - low) / bins
    counts = [0] * bins
    for value in data:
        index = min(int((value - low) / step), bins - 1)
        counts[index] += 1

    left, right, top, bottom = 42, 12, 10, 26
    plot_w, plot_h = width - left - right, height - top - bottom
    peak = max(counts) or 1
    slot = plot_w / bins
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    for i, count in enumerate(counts):
        bucket_low = low + step * i
        colour = negative if bucket_low + step / 2 < 0 else positive
        bar_h = count / peak * plot_h
        parts.append(
            f'<rect x="{left + slot * i:.1f}" y="{top + plot_h - bar_h:.1f}" '
            f'width="{max(slot - 1, 1):.1f}" height="{bar_h:.1f}" fill="{colour}" opacity="0.8"/>'
        )
    if low < 0 < high:
        x = left + (0 - low) / (high - low) * plot_w
        parts.append(
            f'<line class="zero" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}"/>'
        )
    parts.append(f'<text class="tick" x="{left}" y="{height - 8}">{_fmt_number(low)}</text>')
    parts.append(
        f'<text class="tick" x="{width - right}" y="{height - 8}" '
        f'text-anchor="end">{_fmt_number(high)}</text>'
    )
    parts.append(f'<text class="tick" x="{left - 6}" y="{top + 8}" text-anchor="end">{peak}</text>')
    parts.append("</svg>")
    return "".join(parts)


def fold_ribbon(
    folds: Sequence[tuple[int, str, _dt.datetime, _dt.datetime]],
    *,
    width: int = 860,
    row_height: int = 18,
) -> str:
    """The walk-forward structure: one row per fold, coloured by split role.

    Seeing warm-up, train, purge, validation, test and embargo laid out is the quickest
    way to notice a plan that is not what you intended.
    """
    if not folds:
        return '<div class="empty">no folds</div>'
    colours = {
        "warmup": "#9aa5b1",
        "train": "#3b7dd8",
        "purge": "#d9822b",
        "validation": "#8e6fd8",
        "test": "#2f9e68",
        "embargo": "#c0392b",
    }
    indices = sorted({f[0] for f in folds})
    start = min(f[2] for f in folds).timestamp()
    end = max(f[3] for f in folds).timestamp()
    span = (end - start) or 1.0
    left, right = 42, 12
    plot_w = width - left - right
    height = len(indices) * row_height + 30

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    for row, index in enumerate(indices):
        y = row * row_height + 4
        parts.append(
            f'<text class="tick" x="{left - 6}" y="{y + 12}" text-anchor="end">f{index}</text>'
        )
        for fold_index, role, begins, ends in folds:
            if fold_index != index:
                continue
            x0 = left + (begins.timestamp() - start) / span * plot_w
            x1 = left + (ends.timestamp() - start) / span * plot_w
            parts.append(
                f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 1):.1f}" '
                f'height="{row_height - 5}" fill="{colours.get(role, "#888")}" opacity="0.85">'
                f"<title>fold {fold_index} {role}: "
                f"{begins:%Y-%m-%d %H:%M} to {ends:%Y-%m-%d %H:%M}</title></rect>"
            )
    legend = " ".join(
        f'<span class="key"><i style="background:{c}"></i>{r}</span>' for r, c in colours.items()
    )
    parts.append("</svg>")
    return "".join(parts) + f'<div class="legend">{legend}</div>'
