"""cont_v1 internal beta diagnostics with cont_v2 reference.
Replaces plot_betas_figure in plot_diagnostics.py. Three of four panels
plot a different metric per variant, so they use Panel.custom rather than
the declarative engine."""

from __future__ import annotations

from typing import Optional

from masa.plotting.specs import register
from masa.plotting.specs.base import HLine, Panel, PlotSpec, RenderContext
from masa.plotting.config import VariantStyle

_INIT_SAFETY_BOUND = 0.5
_X_LABEL = "Training steps (thousands)"
_STEP_SCALE = 1000.0


def _find(ctx: RenderContext, name: str) -> Optional[VariantStyle]:
    return next((v for v in ctx.variants if v.name == name), None)


def _plot_per_seed(ax, ctx: RenderContext, v: VariantStyle, metric: str,
                   base_linestyle: str = "-") -> None:
    df = ctx.long_df
    df = df[(df["env"] == ctx.env) & (df["variant"] == v.name) & (df["metric"] == metric)]
    if df.empty:
        return
    seeds = sorted(df["seed"].unique())
    for i, seed in enumerate(seeds):
        sub = df[df["seed"] == seed].sort_values("step")
        if sub.empty:
            continue
        steps = sub["step"].values / _STEP_SCALE
        # When base_linestyle is dashed, keep it dashed for all seeds; otherwise
        # distinguish later seeds via dashed style.
        ls = base_linestyle if base_linestyle != "-" else ("-" if i == 0 else "--")
        ax.plot(steps, sub["value"].values, color=v.colour, linewidth=1.6,
                alpha=0.95 if i == 0 else 0.55, linestyle=ls)


def _draw_std_compare(ax, ctx: RenderContext) -> None:
    v1, v2 = _find(ctx, "cont_v1"), _find(ctx, "cont_v2")
    if v1 is not None:
        _plot_per_seed(ax, ctx, v1, "train/stats/betas_std")
    if v2 is not None:
        _plot_per_seed(ax, ctx, v2, "train/stats/mix_std")


def _draw_min_max(ax, ctx: RenderContext) -> None:
    v1 = _find(ctx, "cont_v1")
    if v1 is None:
        return
    _plot_per_seed(ax, ctx, v1, "train/stats/betas_min", base_linestyle="-")
    _plot_per_seed(ax, ctx, v1, "train/stats/betas_max", base_linestyle="--")


def _draw_mean_compare(ax, ctx: RenderContext) -> None:
    v1, v2 = _find(ctx, "cont_v1"), _find(ctx, "cont_v2")
    if v1 is not None:
        _plot_per_seed(ax, ctx, v1, "train/stats/betas_mean")
    if v2 is not None:
        _plot_per_seed(ax, ctx, v2, "train/stats/mix_mean")


betas = register(PlotSpec(
    id="betas",
    grid=(2, 2),
    panels=(
        Panel(metric="train/stats/betas_std",
              title="betas_std (cont_v1)  vs  mix_std (cont_v2)",
              source="per_seed",
              custom=_draw_std_compare,
              xlabel=_X_LABEL,
              variants_filter=("cont_v1", "cont_v2")),
        Panel(metric="train/stats/betas_min",
              title="betas_min (solid) and betas_max (dashed) — cont_v1",
              source="per_seed",
              custom=_draw_min_max,
              hlines=(HLine(0.0), HLine(1.0)),
              xlabel=_X_LABEL,
              variants_filter=("cont_v1",)),
        Panel(metric="train/stats/proj_penalty",
              title="proj_penalty (cont_v1)",
              source="per_seed",
              variants_filter=("cont_v1",),
              xlabel=_X_LABEL),
        Panel(metric="train/stats/betas_mean",
              title="betas_mean (cont_v1)  vs  mix_mean (cont_v2)",
              source="per_seed",
              custom=_draw_mean_compare,
              hlines=(HLine(_INIT_SAFETY_BOUND, style="--", linewidth=0.8, alpha=0.6),),
              xlabel=_X_LABEL,
              variants_filter=("cont_v1", "cont_v2")),
    ),
    suptitle_fmt=(
        "{env_title}  —  cont_v1 Beta Diagnostics  "
        "(yellow = cont_v1, sky blue = cont_v2 reference)  "
        "solid = seed 1, dashed = seed 2+"
    ),
    figsize=(12.0, 9.0),
    out_subdir="betas",
))
