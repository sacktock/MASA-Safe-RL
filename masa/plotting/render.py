"""Generic renderer: consumes PlotSpec + RenderContext, writes one figure per env."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .config import VariantStyle
from .io_utils import ensure_dir, get_logger
from .specs.base import Panel, PlotSpec, RenderContext

STEP_SCALE = 1000.0
DEFAULT_CELL = (3.5, 3.0)

RC_PARAMS = {
    "font.family": "serif",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
}


def apply_rc() -> None:
    plt.rcParams.update(RC_PARAMS)


def render_spec(spec: PlotSpec, ctx: RenderContext, output_dir: Path) -> Optional[Path]:
    log = get_logger()
    env_title = " ".join(w.capitalize() for w in ctx.env.split("_"))
    drawn: set[str] = set()

    if spec.figure_builder is not None:
        figsize = spec.figsize or (10.0, 7.0)
        fig = plt.figure(figsize=figsize)
        fig.suptitle(spec.suptitle_fmt.format(env_title=env_title),
                     fontsize=14, fontweight="bold")
        try:
            spec.figure_builder(fig, ctx, drawn)
        except Exception as e:
            log.warning(f"  {spec.id} / {ctx.env}: figure_builder failed: {e}")
            plt.close(fig)
            return None
        if not drawn:
            log.info(f"  {spec.id} / {ctx.env}: builder produced nothing, skipping.")
            plt.close(fig)
            return None
    else:
        panels = spec.resolve_panels(ctx)
        if not panels:
            log.info(f"  {spec.id} / {ctx.env}: no panels resolved, skipping.")
            return None
        rows, cols = spec.resolve_grid(panels)
        figsize = spec.figsize or (DEFAULT_CELL[0] * cols + 1.5, DEFAULT_CELL[1] * rows + 1.0)
        fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axs_flat = axs.flatten()
        fig.suptitle(spec.suptitle_fmt.format(env_title=env_title),
                     fontsize=14, fontweight="bold")
        for i, panel in enumerate(panels):
            if i >= len(axs_flat):
                log.warning(f"  {spec.id}: panel #{i} ({panel.title}) exceeds grid {rows}x{cols}")
                break
            ax = axs_flat[i]
            try:
                _draw_panel(ax, panel, ctx, drawn)
                _finalise_panel(ax, panel)
            except Exception as e:
                log.warning(f"  {spec.id} / {ctx.env} / panel '{panel.title}': {e}")
                ax.set_visible(False)
        for j in range(len(panels), len(axs_flat)):
            axs_flat[j].set_visible(False)

    if spec.show_legend and drawn:
        _draw_legend(fig, ctx.variants, drawn)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_dir = ensure_dir(output_dir / spec.out_subdir) if spec.out_subdir else ensure_dir(output_dir)
    out_path = out_dir / f"{spec.id}_{ctx.env}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {out_path}")
    return out_path


def draw_panel_into(ax, panel: Panel, ctx: RenderContext, drawn: set[str]) -> None:
    """Public helper for figure_builders that want to draw a Panel into a custom Axes."""
    _draw_panel(ax, panel, ctx, drawn)
    _finalise_panel(ax, panel)


# -------- panel dispatch ---------------------------------------------------

def _draw_panel(ax, panel: Panel, ctx: RenderContext, drawn: set[str]) -> None:
    if panel.custom is not None:
        panel.custom(ax, ctx)
        for v in _filter_variants(panel, ctx):
            drawn.add(v.name)
        return
    if panel.source == "aggregated":
        _draw_aggregated(ax, panel, ctx, drawn)
    elif panel.source == "per_seed":
        _draw_per_seed(ax, panel, ctx, drawn)
    elif panel.source == "logged_quantile":
        _draw_logged_quantile(ax, panel, ctx, drawn)
    else:
        raise ValueError(f"unknown panel source: {panel.source!r}")


def _draw_aggregated(ax, panel: Panel, ctx: RenderContext, drawn: set[str]) -> None:
    qdf = ctx.quantile_df
    qdf = qdf[(qdf["env"] == ctx.env) & (qdf["metric"] == panel.metric)]
    if qdf.empty:
        return
    for v in _filter_variants(panel, ctx):
        sub = qdf[qdf["variant"] == v.name].sort_values("step")
        if sub.empty:
            continue
        steps = sub["step"].values / STEP_SCALE
        central = _maybe_smooth(sub[panel.central].values, panel.smooth_alpha)
        ax.plot(steps, central, color=v.colour, linewidth=2.0)
        for band in panel.bands:
            if band.lower in sub.columns and band.upper in sub.columns:
                lo = _maybe_smooth(sub[band.lower].values, panel.smooth_alpha)
                hi = _maybe_smooth(sub[band.upper].values, panel.smooth_alpha)
                ax.fill_between(steps, lo, hi, color=v.colour, alpha=band.alpha, linewidth=0)
        drawn.add(v.name)


def _draw_per_seed(ax, panel: Panel, ctx: RenderContext, drawn: set[str]) -> None:
    df = ctx.long_df
    df = df[(df["env"] == ctx.env) & (df["metric"] == panel.metric)]
    if df.empty:
        return
    for v in _filter_variants(panel, ctx):
        sub_v = df[df["variant"] == v.name]
        if sub_v.empty:
            continue
        seeds = sorted(sub_v["seed"].unique())
        for i, seed in enumerate(seeds):
            sub = sub_v[sub_v["seed"] == seed].sort_values("step")
            if sub.empty:
                continue
            steps = sub["step"].values / STEP_SCALE
            values = _maybe_smooth(sub["value"].values, panel.smooth_alpha)
            ax.plot(steps, values, color=v.colour, linewidth=1.6,
                    alpha=0.95 if i == 0 else 0.55,
                    linestyle="-" if i == 0 else "--")
        drawn.add(v.name)


def _draw_logged_quantile(ax, panel: Panel, ctx: RenderContext, drawn: set[str]) -> None:
    seed = ctx.seed_for_logged_quantiles
    df = ctx.long_df
    df = df[(df["env"] == ctx.env) & (df["seed"] == seed)]
    if df.empty:
        return
    base = panel.metric
    for v in _filter_variants(panel, ctx):
        sub_v = df[df["variant"] == v.name]
        if sub_v.empty:
            continue
        central_col = f"{base}_{panel.central}"
        central_sub = sub_v[sub_v["metric"] == central_col].sort_values("step")
        if central_sub.empty:
            continue
        steps = central_sub["step"].values / STEP_SCALE
        central_vals = _maybe_smooth(central_sub["value"].values, panel.smooth_alpha)
        ax.plot(steps, central_vals, color=v.colour, linewidth=1.6)
        for band in panel.bands:
            lo_col = f"{base}_{band.lower}"
            hi_col = f"{base}_{band.upper}"
            lo_sub = sub_v[sub_v["metric"] == lo_col][["step", "value"]]
            hi_sub = sub_v[sub_v["metric"] == hi_col][["step", "value"]]
            if lo_sub.empty or hi_sub.empty:
                continue
            merged = lo_sub.merge(hi_sub, on="step", suffixes=("_lo", "_hi")).sort_values("step")
            if merged.empty:
                continue
            s = merged["step"].values / STEP_SCALE
            lo = _maybe_smooth(merged["value_lo"].values, panel.smooth_alpha)
            hi = _maybe_smooth(merged["value_hi"].values, panel.smooth_alpha)
            ax.fill_between(s, lo, hi, color=v.colour, alpha=band.alpha, linewidth=0)
        drawn.add(v.name)


# -------- finishing touches ------------------------------------------------

def _finalise_panel(ax, panel: Panel) -> None:
    ax.set_title(panel.title, fontsize=11, pad=5)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel, fontsize=10)
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel, fontsize=10)
    if panel.ylim is not None:
        ax.set_ylim(*panel.ylim)
    for hl in panel.hlines:
        ax.axhline(hl.y, color=hl.colour, linestyle=hl.style,
                   linewidth=hl.linewidth, alpha=hl.alpha)
    ax.set_xlim(left=0)
    ax.tick_params(axis="x", direction="in", top=True)
    ax.tick_params(axis="y", direction="in", right=True)


def _draw_legend(fig, variants: Sequence[VariantStyle], drawn: set[str]) -> None:
    ordered = [v for v in sorted(variants, key=lambda x: x.order) if v.name in drawn]
    handles = [Line2D([0], [0], color=v.colour, linewidth=2.0) for v in ordered]
    labels = [v.label for v in ordered]
    fig.legend(handles, labels,
               loc="lower center", ncol=min(4, max(1, len(ordered))),
               bbox_to_anchor=(0.5, -0.02),
               fontsize=11, handlelength=2.5)


def _filter_variants(panel: Panel, ctx: RenderContext) -> list[VariantStyle]:
    if panel.variants_filter is None:
        return list(ctx.variants)
    keep = set(panel.variants_filter)
    return [v for v in ctx.variants if v.name in keep]


def _maybe_smooth(values, alpha):
    if alpha is None or len(values) == 0:
        return values
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean().values
