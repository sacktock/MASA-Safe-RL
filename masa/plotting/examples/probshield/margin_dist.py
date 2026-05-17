"""Margin distribution: per-horizon × per-variant logged quantile bands (top)
plus aggregated tail fractions (bottom). Uses figure_builder because the
layout spans columns via GridSpec.

Replaces plot_margin_distribution_figure in plot_diagnostics.py."""

from __future__ import annotations

import re

from masa.plotting.specs import register
from masa.plotting.specs.base import Band, HLine, Panel, PlotSpec, RenderContext
from .margin import _discover_horizons  # share horizon discovery

_X_LABEL = "Training steps (thousands)"
_LOGGED_BANDS = (Band("q05", "q95", alpha=0.15), Band("q25", "q75", alpha=0.35))
_LIMIT_HLINES = (HLine(0.0), HLine(1.0))


def _build(fig, ctx: RenderContext, drawn: set[str]) -> None:
    from masa.plotting.render import draw_panel_into

    horizons = _discover_horizons(ctx)
    if not horizons:
        return

    variants = list(ctx.variants)
    n_h = len(horizons)
    n_v = len(variants)

    gs = fig.add_gridspec(
        n_h + 1, n_v,
        height_ratios=[1.0] * n_h + [1.3],
    )

    # Top grid: one cell per (horizon, variant); logged quantile bands, seed 1.
    for row_i, t in enumerate(horizons):
        for col_i, v in enumerate(variants):
            ax = fig.add_subplot(gs[row_i, col_i])
            panel = Panel(
                metric=f"train/stats/margin_{t}",
                title=v.label if row_i == 0 else "",
                source="logged_quantile",
                bands=_LOGGED_BANDS,
                hlines=_LIMIT_HLINES,
                ylim=(-0.05, 1.05),
                variants_filter=(v.name,),
                ylabel=rf"$q_{{{t}}}$" if col_i == 0 else None,
                xlabel=_X_LABEL if row_i == n_h - 1 else None,
            )
            draw_panel_into(ax, panel, ctx, drawn)

    # Bottom row: two spanning subplots for tail fractions; all variants overlaid.
    if n_v >= 2:
        half = max(1, n_v // 2)
        tail_specs = [
            (fig.add_subplot(gs[n_h, :half]),
             "train/stats/margin_frac_near_0",
             r"fraction of steps with $q_t < 0.005$"),
            (fig.add_subplot(gs[n_h, half:]),
             "train/stats/margin_frac_near_1",
             r"fraction of steps with $q_t > 0.995$"),
        ]
    else:
        tail_specs = [
            (fig.add_subplot(gs[n_h, 0]),
             "train/stats/margin_frac_near_0",
             r"fraction of steps with $q_t < 0.005$"),
        ]
    for ax, metric, title in tail_specs:
        panel = Panel(
            metric=metric,
            title=title,
            source="per_seed",
            ylim=(-0.02, 1.02),
            xlabel=_X_LABEL,
        )
        draw_panel_into(ax, panel, ctx, drawn)


margin_dist = register(PlotSpec(
    id="margin_dist",
    figure_builder=_build,
    suptitle_fmt=(
        "{env_title}  —  Margin Distribution (q05-q95 light, q25-q75 dark, q50 line)"
        " + Global Tail Fractions (seed 1)"
    ),
    figsize=(14.0, 11.0),
    out_subdir="margin_dist",
))
