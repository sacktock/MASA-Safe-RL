from __future__ import annotations

from html import escape
from typing import Mapping, Sequence


def _obs_orig(row: Mapping[str, object]) -> int:
    return int(row["obs_orig"])


def _labels_text(labels: object) -> str:
    if not labels:
        return "none"
    if isinstance(labels, str):
        return labels
    return ", ".join(str(label) for label in labels)


def _metric(row: Mapping[str, object], group: str, key: str, default: float = 0.0) -> float:
    metrics = row.get(group)
    if not isinstance(metrics, Mapping):
        return default
    value = metrics.get(key, default)
    return float(value)


def _state_center(state: int, grid_size: int, cell: int, margin: int) -> tuple[int, int]:
    row, col = divmod(int(state), int(grid_size))
    return margin + col * cell + cell // 2, margin + row * cell + cell // 2


def _state_label(visits: Sequence[int]) -> str:
    if len(visits) == 1:
        return str(visits[0])
    if visits == list(range(visits[0], visits[-1] + 1)):
        return f"{visits[0]}-{visits[-1]}"
    return ",".join(str(v) for v in visits)


def render_colour_bomb_trace_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
    grid_size: int,
    start_states: Sequence[int],
    wall_states: Sequence[int],
    bomb_states: Sequence[int],
    goal_states: Sequence[int] = (),
    medic_states: Sequence[int] = (),
) -> str:
    cell = 34 if grid_size <= 10 else 24
    margin = 42
    legend_width = 190
    grid_px = int(grid_size) * cell
    width = margin * 2 + grid_px + legend_width
    height = margin * 2 + grid_px

    states = [_obs_orig(row) for row in rows]
    visits: dict[int, list[int]] = {}
    for idx, state in enumerate(states):
        visits.setdefault(state, []).append(idx)

    def rect_for(state: int, fill: str, stroke: str = "#d1d5db") -> str:
        row, col = divmod(int(state), int(grid_size))
        x = margin + col * cell
        y = margin + row * cell
        return (
            f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        "<defs>",
        '<marker id="trace-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L7,3 z" fill="#1f2937"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="18" y="25" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
    ]

    for row in range(grid_size):
        for col in range(grid_size):
            x = margin + col * cell
            y = margin + row * cell
            fill = "#f8fafc" if (row + col) % 2 == 0 else "#eef2f7"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#d1d5db" stroke-width="1"/>'
            )

    for state in wall_states:
        parts.append(rect_for(state, "#334155", "#1f2937"))
    for state in goal_states:
        parts.append(rect_for(state, "#bfdbfe", "#60a5fa"))
    for state in medic_states:
        parts.append(rect_for(state, "#bbf7d0", "#22c55e"))
    for state in bomb_states:
        parts.append(rect_for(state, "#111827", "#000000"))
    for state in start_states:
        parts.append(rect_for(state, "#fde68a", "#f59e0b"))

    if len(states) > 1:
        points = [
            _state_center(state, grid_size, cell, margin)
            for state in states
        ]
        d = " ".join(f"L{x},{y}" if idx else f"M{x},{y}" for idx, (x, y) in enumerate(points))
        parts.append(f'<path d="{d}" fill="none" stroke="#1f2937" stroke-width="3" stroke-linejoin="round" marker-end="url(#trace-arrow)"/>')

    for state, indices in visits.items():
        cx, cy = _state_center(state, grid_size, cell, margin)
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{max(9, cell // 3)}" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>')
        parts.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-family="sans-serif" font-size="{10 if len(indices) > 1 else 12}" font-weight="700" fill="#ffffff">{escape(_state_label(indices))}</text>'
        )

    legend_x = margin + grid_px + 28
    legend = [
        ("start", "#fde68a"),
        ("bomb", "#111827"),
        ("goal", "#bfdbfe"),
        ("medic", "#bbf7d0"),
        ("wall", "#334155"),
        ("visited", "#2563eb"),
    ]
    parts.append(f'<text x="{legend_x}" y="{margin}" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">Legend</text>')
    for idx, (label, color) in enumerate(legend, start=1):
        y = margin + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" rx="2" fill="{color}" stroke="#9ca3af"/>')
        parts.append(f'<text x="{legend_x + 22}" y="{y}" font-family="sans-serif" font-size="12" fill="#374151">{label}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def render_dfa_diagram_svg(
    *,
    title: str,
    states: Sequence[int],
    accepting: Sequence[int],
    edges: Sequence[tuple[int, int, str]],
    state_labels: Mapping[int, str] | None = None,
) -> str:
    state_labels = dict(state_labels or {})
    radius = 26
    gap = 115
    margin = 44
    width = max(440, margin * 2 + gap * max(1, len(states) - 1) + radius * 2)
    height = 250
    y = 122
    content_width = gap * max(1, len(states) - 1) + radius * 2
    left = (width - content_width) / 2
    x_by_state = {state: left + radius + idx * gap for idx, state in enumerate(states)}
    accepting_set = set(accepting)
    start_x = x_by_state[states[0]] - radius

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        "<defs>",
        '<marker id="dfa-arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L8,3 z" fill="#374151"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="18" y="26" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<line x1="{start_x - 22:.1f}" y1="{y}" x2="{start_x - 1:.1f}" y2="{y}" stroke="#374151" stroke-width="2.5" marker-end="url(#dfa-arrow)"/>',
    ]

    for source, target, label in edges:
        sx = x_by_state[source]
        tx = x_by_state[target]
        if source == target:
            parts.append(
                f'<path d="M{sx - 10} {y - radius} C{sx - 42} {y - 70}, {sx + 42} {y - 70}, {sx + 10} {y - radius}" fill="none" stroke="#374151" stroke-width="2" marker-end="url(#dfa-arrow)"/>'
            )
            parts.append(f'<text x="{sx}" y="{y - 72}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#374151">{escape(label)}</text>')
        else:
            top = y - 46 if x_by_state[source] < x_by_state[target] else y + 52
            parts.append(
                f'<path d="M{sx + radius} {y} C{(sx + tx) // 2} {top}, {(sx + tx) // 2} {top}, {tx - radius} {y}" fill="none" stroke="#374151" stroke-width="2" marker-end="url(#dfa-arrow)"/>'
            )
            parts.append(f'<text x="{(sx + tx) // 2}" y="{top - 6 if top < y else top + 16}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#111827">{escape(label)}</text>')

    for state in states:
        x = x_by_state[state]
        fill = "#fee2e2" if state in accepting_set else "#dcfce7"
        stroke = "#991b1b" if state in accepting_set else "#166534"
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        if state in accepting_set:
            parts.append(f'<circle cx="{x}" cy="{y}" r="{radius - 7}" fill="none" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{x}" y="{y - 3}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">q{state}</text>')
        parts.append(f'<text x="{x}" y="{y + 15}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#374151">{escape(state_labels.get(state, ""))}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def render_ltl_rollout_timeline_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
) -> str:
    cell = 72
    left = 24
    width = max(760, left * 2 + cell * len(rows))
    height = 178
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="18" y="26" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<line x1="{left + 24}" y1="82" x2="{left + cell * len(rows) - 24}" y2="82" stroke="#9ca3af" stroke-width="2"/>',
    ]

    for idx, row in enumerate(rows):
        x = left + idx * cell
        labels = list(row.get("labels", []))
        violation = _metric(row, "constraint_step", "violation")
        fill = "#fee2e2" if violation >= 0.5 else "#fff7ed" if "bomb" in labels else "#eff6ff"
        stroke = "#ef4444" if violation >= 0.5 else "#fdba74" if "bomb" in labels else "#93c5fd"
        label_text = _labels_text(labels)
        if len(label_text) > 10:
            label_text = label_text[:9] + "."
        parts.append(f'<rect x="{x}" y="50" width="58" height="88" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + 29}" y="70" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#111827">t{idx}</text>')
        parts.append(f'<text x="{x + 29}" y="90" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#374151">s={_obs_orig(row)}</text>')
        parts.append(f'<text x="{x + 29}" y="108" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#374151">q={int(row.get("automaton_state", 0))}</text>')
        parts.append(f'<text x="{x + 29}" y="126" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#374151">{escape(label_text)}</text>')
        if violation >= 0.5:
            parts.append(f'<text x="{x + 29}" y="156" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#991b1b">violation</text>')

    parts.append("</svg>")
    return "\n".join(parts)
