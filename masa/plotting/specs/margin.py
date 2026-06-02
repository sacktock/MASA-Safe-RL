"""Margin figure: per-horizon fan charts of the shield's safety budget.

Each panel shows one margin horizon t (auto-discovered from the long-form
columns). Within a panel, every variant is drawn as a median line with two
nested shaded bands (q05/q95 outer, q25/q75 inner).

Two data sources are supported, picked automatically per env:

* ``aggregated`` (current sweeps): the wrapper logs raw scalars
  ``train/stats/margin_<t>``; we aggregate across seeds via the project's
  quantile frame. Use this when you have multiple seeds per variant.

* ``logged_quantile`` (older runs): W&B derived per-quantile scalars
  ``train/stats/margin_<t>_q05`` ... ``_q95`` from a ``wandb.Histogram``
  payload. One seed is plotted directly.
"""

from __future__ import annotations

import re
from typing import Sequence

from . import register
from .base import Band, Panel, PlotSpec, RenderContext

_RAW_RE = re.compile(r"^train/stats/margin_(\d+)$")
_LOGGED_Q_RE = re.compile(r"^train/stats/margin_(\d+)_q50$")
_OUTER_BAND = Band("q05", "q95", alpha=0.15)
_INNER_BAND = Band("q25", "q75", alpha=0.35)
_X_LABEL = "Training steps (in thousands)"


def _discover_horizons(ctx: RenderContext, pattern: re.Pattern) -> list[int]:
    metrics = ctx.long_df.loc[ctx.long_df["env"] == ctx.env, "metric"].unique()
    return sorted({
        int(m.group(1))
        for c in metrics
        for m in [pattern.match(c)]
        if m is not None
    })


def _build_panels(ctx: RenderContext) -> Sequence[Panel]:
    # Prefer aggregated mode (real seed variability). Fall back to the
    # logged-quantile mode for legacy runs that only ship the W&B-derived
    # per-quantile scalars.
    horizons = _discover_horizons(ctx, _RAW_RE)
    if horizons:
        source = "aggregated"
        metric_for = lambda t: f"train/stats/margin_{t}"
    else:
        horizons = _discover_horizons(ctx, _LOGGED_Q_RE)
        if not horizons:
            return ()
        source = "logged_quantile"
        metric_for = lambda t: f"train/stats/margin_{t}"

    cols = min(3, len(horizons))
    last_row_start = len(horizons) - cols
    return tuple(
        Panel(
            metric=metric_for(t),
            title=f"Margin at horizon t={t}",
            source=source,
            bands=(_OUTER_BAND, _INNER_BAND),
            ylabel="Safety budget q_t" if i % cols == 0 else None,
            xlabel=_X_LABEL if i >= last_row_start else None,
        )
        for i, t in enumerate(horizons)
    )


def _grid_for(panels: Sequence[Panel]) -> tuple[int, int]:
    n = len(panels)
    if n == 0:
        return (1, 1)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    return (rows, cols)


margin = register(PlotSpec(
    id="margin",
    panels=_build_panels,
    grid=_grid_for,
    suptitle_fmt="{env_title} — Shield Safety Budget over Training",
    out_subdir="margin",
))
