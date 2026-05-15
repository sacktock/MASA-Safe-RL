"""Visual helpers for the constraints tour tutorial."""

from __future__ import annotations

from html import escape
from typing import Mapping, Sequence

from masa.envs.tabular.colour_grid_world import (
    BLUE_STATE,
    GOAL_STATE,
    GREEN_STATE,
    GRID_SIZE,
    PURPLE_STATE,
    START_STATE,
)

CELL_SIZE = 42
PADDING = 34
SPECIAL_STATES = {
    START_STATE: ("start", "#dbeafe"),
    BLUE_STATE: ("blue", "#93c5fd"),
    GREEN_STATE: ("green", "#bbf7d0"),
    PURPLE_STATE: ("purple", "#ddd6fe"),
    GOAL_STATE: ("goal", "#dcfce7"),
}


def state_center(state: int) -> tuple[float, float]:
    row, col = divmod(int(state), GRID_SIZE)
    return (
        PADDING + col * CELL_SIZE + CELL_SIZE / 2,
        PADDING + row * CELL_SIZE + CELL_SIZE / 2,
    )


def render_grid_trace_svg(rows: Sequence[Mapping[str, object]], title: str) -> str:
    states = [int(row["obs"]) for row in rows]
    width = PADDING * 2 + GRID_SIZE * CELL_SIZE
    height = PADDING * 2 + GRID_SIZE * CELL_SIZE + 46
    points = " ".join(
        f"{x:.1f},{y:.1f}" for state in states for x, y in [state_center(state)]
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{PADDING}" y="24" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
    ]

    for state in range(GRID_SIZE * GRID_SIZE):
        row, col = divmod(state, GRID_SIZE)
        x = PADDING + col * CELL_SIZE
        y = PADDING + row * CELL_SIZE
        label, fill = SPECIAL_STATES.get(state, ("", "#f9fafb"))
        parts.append(
            f'<rect x="{x}" y="{y}" width="{CELL_SIZE}" height="{CELL_SIZE}" fill="{fill}" stroke="#d1d5db"/>'
        )
        if label:
            parts.append(
                f'<text x="{x + 4}" y="{y + CELL_SIZE - 6}" font-family="sans-serif" font-size="9" fill="#374151">{label}</text>'
            )

    if len(states) > 1:
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#111827" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" opacity="0.72"/>'
        )

    for step, state in enumerate(states):
        x, y = state_center(state)
        is_final = step == len(states) - 1
        fill = (
            "#b91c1c"
            if state == BLUE_STATE
            else "#166534"
            if state == GOAL_STATE
            else "#111827"
        )
        radius = 12 if is_final else 10
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#ffffff">{step}</text>'
        )

    legend_y = PADDING + GRID_SIZE * CELL_SIZE + 28
    parts.extend(
        [
            f'<text x="{PADDING}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="#374151">Numbers are reset/step indices from the actual seeded rollout.</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def render_dfa_svg() -> str:
    return """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 190" width="520" height="190" role="img" aria-label="Never blue DFA">
  <defs>
    <marker id="dfa-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#111827" />
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="28" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">Never-blue DFA used by ltl_safety</text>
  <line x1="55" y1="96" x2="91" y2="96" stroke="#111827" stroke-width="2.5" marker-end="url(#dfa-arrow)"/>
  <circle cx="140" cy="96" r="40" fill="#dcfce7" stroke="#166534" stroke-width="3"/>
  <text x="140" y="91" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#14532d">q0</text>
  <text x="140" y="110" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#14532d">safe</text>
  <circle cx="380" cy="96" r="42" fill="#fee2e2" stroke="#991b1b" stroke-width="3"/>
  <circle cx="380" cy="96" r="34" fill="none" stroke="#991b1b" stroke-width="2"/>
  <text x="380" y="91" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#7f1d1d">q1</text>
  <text x="380" y="110" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#7f1d1d">unsafe</text>
  <path d="M180 96 C235 52 285 52 338 96" fill="none" stroke="#111827" stroke-width="2.5" marker-end="url(#dfa-arrow)"/>
  <text x="260" y="55" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">blue</text>
  <path d="M122 58 C86 25 190 25 158 58" fill="none" stroke="#166534" stroke-width="2" marker-end="url(#dfa-arrow)"/>
  <text x="140" y="24" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#166534">implicit not blue loop</text>
  <path d="M362 58 C326 25 430 25 398 58" fill="none" stroke="#991b1b" stroke-width="2" marker-end="url(#dfa-arrow)"/>
  <text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#991b1b">implicit loop after violation</text>
</svg>
""".strip()


def render_constraint_semantics_svg() -> str:
    boxes = [
        ("cmdp", "sum cost <= budget"),
        ("prob", "unsafe fraction <= alpha"),
        ("pctl", "Pr(formula satisfied) >= threshold"),
        ("reach_avoid", "reach goal before blue"),
        ("ltl_safety", "DFA avoids accepting unsafe state"),
    ]
    width = 760
    height = 310
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Constraint semantics diagram">',
        '<defs><marker id="sem-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#4b5563" /></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">Same labels, different constraint semantics</text>',
        '<rect x="26" y="86" width="160" height="88" rx="8" fill="#f3f4f6" stroke="#9ca3af"/>',
        '<text x="106" y="117" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">labels</text>',
        '<text x="106" y="140" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">blue, goal, ...</text>',
    ]
    for index, (name, meaning) in enumerate(boxes):
        y = 58 + index * 46
        parts.append(
            f'<line x1="186" y1="130" x2="270" y2="{y + 20}" stroke="#4b5563" stroke-width="1.8" marker-end="url(#sem-arrow)"/>'
        )
        parts.append(
            f'<rect x="280" y="{y}" width="430" height="36" rx="7" fill="#eff6ff" stroke="#93c5fd"/>'
        )
        parts.append(
            f'<text x="300" y="{y + 23}" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2937">{escape(name)}</text>'
        )
        parts.append(
            f'<text x="420" y="{y + 23}" font-family="sans-serif" font-size="12" fill="#374151">{escape(meaning)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)
