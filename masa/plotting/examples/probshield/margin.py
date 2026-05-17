"""Margin trajectories: 2 rows (mean, std) x N cols (discovered horizons),
per-seed lines for every variant. Replaces plot_margin_figure in
plot_diagnostics.py."""

from __future__ import annotations

import re
from typing import Sequence

from masa.plotting.specs import register
from masa.plotting.specs.base import Panel, PlotSpec, RenderContext

_HORIZON_RE = re.compile(r"^train/stats/margin_(\d+)_mean$")
_X_LABEL = "Training steps (thousands)"


def _discover_horizons(ctx: RenderContext) -> list[int]:
    df = ctx.long_df
    df = df[df["env"] == ctx.env]
    horizons: set[int] = set()
    for m in df["metric"].unique():
        match = _HORIZON_RE.match(m)
        if match:
            horizons.add(int(match.group(1)))
    return sorted(horizons)


def _panels(ctx: RenderContext) -> Sequence[Panel]:
    horizons = _discover_horizons(ctx)
    if not horizons:
        return ()
    panels: list[Panel] = []
    # Row 0: means across horizons
    for t in horizons:
        panels.append(Panel(
            metric=f"train/stats/margin_{t}_mean",
            title=rf"$q_{{{t}}}$ mean",
            source="per_seed",
        ))
    # Row 1: stds across horizons
    for t in horizons:
        panels.append(Panel(
            metric=f"train/stats/margin_{t}_std",
            title=rf"$q_{{{t}}}$ std",
            source="per_seed",
            xlabel=_X_LABEL,
        ))
    return panels


def _grid(panels: Sequence[Panel]) -> tuple[int, int]:
    return (2, len(panels) // 2) if panels else (1, 1)


margin = register(PlotSpec(
    id="margin",
    panels=_panels,
    grid=_grid,
    suptitle_fmt="{env_title}  —  Margin Trajectories (solid = seed 1, dashed = seed 2+)",
    out_subdir="margin",
))
