from __future__ import annotations

from html import escape
from typing import Mapping, Sequence


def markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    """Render a small Markdown table for tutorial display."""
    if not rows:
        return ""

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3g}"
            values.append(escape(str(value)))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def algorithm_state_summary(name: str, algo: object) -> dict[str, object]:
    """Return compact diagnostics that are stable enough for a smoke tutorial."""
    row: dict[str, object] = {"algorithm": name}
    q_table = getattr(algo, "Q", None)
    if q_table is not None:
        row["Q shape"] = getattr(q_table, "shape", "")
        row["max Q"] = float(q_table.max())
        row["min Q"] = float(q_table.min())

    if hasattr(algo, "D"):
        row["D shape"] = getattr(algo.D, "shape", "")
    if hasattr(algo, "C"):
        row["C shape"] = getattr(algo.C, "shape", "")
    if hasattr(algo, "B"):
        row["B shape"] = getattr(algo.B, "shape", "")
    if hasattr(algo, "S"):
        row["S shape"] = getattr(algo.S, "shape", "")
    return row


def smoke_run_note() -> str:
    return (
        "These are smoke-run diagnostics from tiny training runs. "
        "They prove that the algorithms, environments, and logs connect; "
        "they are not benchmark rankings."
    )
