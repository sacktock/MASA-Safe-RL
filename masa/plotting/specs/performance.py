"""Performance figure: 2x2 grid of reward + satisfaction, raw + EMA-smoothed."""

from __future__ import annotations

from . import register
from .base import Band, Panel, PlotSpec

_REWARD = "eval/rollout/ep_reward"
_SATISFIED = "eval/rollout/satisfied"
_BAND = (Band("q25", "q75"),)
_SMOOTH = 0.3
_X_LABEL = "Training steps (in thousands)"

performance = register(PlotSpec(
    id="performance",
    grid=(2, 2),
    panels=(
        Panel(metric=_REWARD,    title="Cumulative Reward",
              bands=_BAND, ylabel="Avg. Reward per Step"),
        Panel(metric=_SATISFIED, title="Constraint Satisfaction Rate",
              bands=_BAND, ylabel="Avg. Satisfied per Step"),
        Panel(metric=_REWARD,    title="Smoothed Cumulative Reward (EMA α=0.3)",
              bands=_BAND, smooth_alpha=_SMOOTH,
              ylabel="Avg. Reward per Step", xlabel=_X_LABEL),
        Panel(metric=_SATISFIED, title="Smoothed Constraint Satisfaction Rate (EMA α=0.3)",
              bands=_BAND, smooth_alpha=_SMOOTH,
              ylabel="Avg. Satisfied per Step", xlabel=_X_LABEL),
    ),
    figsize=(11.0, 8.5),
    out_subdir="performance",
))
