"""Seaborn plotting utilities for MASA training metrics."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


def smooth(values: np.ndarray, weight: float) -> np.ndarray:
    """Exponential moving average smoothing.

    Args:
        values: 1-D array of scalars.
        weight: Smoothing factor in ``[0, 1)``.  Higher = smoother.

    Returns:
        Smoothed array of the same length.
    """
    smoothed = np.empty_like(values, dtype=float)
    last = float(values[0])
    for i, v in enumerate(values):
        last = last * weight + (1.0 - weight) * float(v)
        smoothed[i] = last
    return smoothed


_RC_PARAMS: Dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "text.usetex": False,
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}


def _thousands_formatter() -> ticker.FuncFormatter:
    """Format tick values as thousands with comma separators."""
    return ticker.FuncFormatter(
        lambda x, _: f"{int(x / 1_000):,}" if x >= 1_000 else f"{int(x)}"
    )


def plot_metrics(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    smooth_weight: float = 0.6,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 5),
    show_legend: bool = True,
    show_axes: bool = True,
    dpi: int = 300,
    hue: str = "run",
    palette: Optional[Dict[str, str]] = None,
    errorbar: Tuple[str, float] = ("sd", 1),
) -> plt.Figure:
    """Plot one or more metrics from a normalised DataFrame.

    Args:
        df: DataFrame with columns ``[step, metric, value, run]``.
        metrics: Which metric names to plot.  ``None`` plots all found in *df*.
        smooth_weight: EMA weight applied per ``(run, metric)`` group.
        title: Optional suptitle for the figure.
        output_path: If given, save the figure to this path.
        figsize: ``(width, height)`` per subplot panel.
        show_legend: Whether to render a shared legend below the plots.
        show_axes: Whether to label axes.
        dpi: Resolution for saved figures.
        hue: Column used to distinguish lines (default ``"run"``).
        palette: Optional colour mapping ``{hue_value: colour}``.
        errorbar: Seaborn errorbar specification for shaded variance bands
            (default ``("sd", 1)`` for +/- 1 standard deviation).

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    with plt.rc_context(_RC_PARAMS):
        return _plot_metrics_inner(
            df, metrics=metrics, smooth_weight=smooth_weight,
            title=title, output_path=output_path, figsize=figsize,
            show_legend=show_legend, show_axes=show_axes, dpi=dpi,
            hue=hue, palette=palette, errorbar=errorbar,
        )


def plot_quantile_metrics(
    q_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    smooth_weight: float = 0.6,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 5),
    show_legend: bool = True,
    show_axes: bool = True,
    dpi: int = 300,
    palette: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Plot q50 ± q25/q75 bands from a quantile DataFrame.

    Args:
        q_df: DataFrame with columns ``[variant, step, metric, q25, q50, q75]``
            as produced by :func:`~masa.plotting.processing.build_quantile_frame`.
        metrics: Metric keys to plot.  ``None`` plots all found in *q_df*.
        smooth_weight: EMA weight applied per ``(variant, metric)`` group.
        title: Per-panel title override.  ``None`` uses the metric name.
        output_path: Save path.  ``None`` skips saving.
        figsize: ``(width, height)`` per subplot panel.
        show_legend: Render a shared legend below the panels.
        show_axes: Label axes.
        dpi: Resolution for saved figures.
        palette: Colour mapping ``{variant: colour}``.  ``None`` uses colorblind.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    with plt.rc_context(_RC_PARAMS):
        return _plot_quantile_inner(
            q_df, metrics=metrics, smooth_weight=smooth_weight,
            title=title, output_path=output_path, figsize=figsize,
            show_legend=show_legend, show_axes=show_axes, dpi=dpi,
            palette=palette,
        )


def _plot_metrics_inner(
    df: pd.DataFrame,
    *,
    metrics: Optional[List[str]],
    smooth_weight: float,
    title: Optional[str],
    output_path: Optional[str],
    figsize: Tuple[int, int],
    show_legend: bool,
    show_axes: bool,
    dpi: int,
    hue: str,
    palette: Optional[Dict[str, str]],
    errorbar: Tuple[str, float],
) -> plt.Figure:
    if metrics is None:
        metrics = sorted(df["metric"].unique().tolist())
    if not metrics:
        raise ValueError("No metrics to plot (empty DataFrame or metrics list).")
    if df.empty:
        raise ValueError("The provided DataFrame is empty. Check if the specified metrics or runs exist.")

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]

    # Apply smoothing per (run, metric) group.
    parts: list[pd.DataFrame] = []
    for (_run, _metric), grp in df.groupby([hue, "metric"]):
        grp = grp.sort_values("step").copy()
        grp["value"] = smooth(grp["value"].values, smooth_weight)
        parts.append(grp)
    df_smooth = pd.concat(parts, ignore_index=True)

    for k, metric in enumerate(metrics):
        ax = axes[k]
        sub = df_smooth[df_smooth["metric"] == metric]
        if sub.empty:
            ax.set_title(metric)
            continue

        plot_kwargs: dict = dict(
            data=sub, x="step", y="value", hue=hue,
            errorbar=errorbar,
            ax=ax, linewidth=2.0, legend=False,
        )
        if palette:
            plot_kwargs["palette"] = palette
        sns.lineplot(**plot_kwargs)

        ax.set_title(_pretty_name(metric) if title is None else title)

        if show_axes:
            ax.set_ylabel(_pretty_name(metric))
            ax.set_xlabel("Training steps (in thousands)")
            ax.xaxis.set_major_formatter(_thousands_formatter())
        else:
            ax.set_ylabel("")
            ax.set_xlabel("")

        ax.tick_params(axis="both", which="both", top=True, right=True)

    # Shared legend below all panels.
    if show_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, 0.02), ncol=min(len(handles), 6),
                frameon=True, edgecolor="white", fancybox=False,
                columnspacing=1.5, handlelength=2.0,
            )

    fig.tight_layout(rect=[0, 0.08, 1, 1] if show_legend else [0, 0, 1, 1])

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)

    return fig


def _plot_quantile_inner(
    q_df: pd.DataFrame,
    *,
    metrics: Optional[List[str]],
    smooth_weight: float,
    title: Optional[str],
    output_path: Optional[str],
    figsize: Tuple[int, int],
    show_legend: bool,
    show_axes: bool,
    dpi: int,
    palette: Optional[Dict[str, str]],
) -> plt.Figure:
    if metrics is None:
        metrics = sorted(q_df["metric"].unique().tolist())
    if not metrics:
        raise ValueError("No metrics to plot.")
    if q_df.empty:
        raise ValueError("The provided DataFrame is empty.")

    variants = sorted(q_df["variant"].unique().tolist())
    if palette is None:
        colours: Dict[str, str] = dict(zip(variants, sns.color_palette("colorblind", len(variants))))
    else:
        colours = {v: palette.get(v, "#888888") for v in variants}

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]

    for k, metric in enumerate(metrics):
        ax = axes[k]
        sub = q_df[q_df["metric"] == metric]
        if sub.empty:
            ax.set_title(metric)
            continue
        for variant in variants:
            vsub = sub[sub["variant"] == variant].sort_values("step")
            if vsub.empty:
                continue
            colour = colours.get(variant)
            steps = vsub["step"].values
            q50 = smooth(vsub["q50"].values, smooth_weight)
            q25 = smooth(vsub["q25"].values, smooth_weight)
            q75 = smooth(vsub["q75"].values, smooth_weight)
            ax.plot(steps, q50, color=colour, linewidth=2.0, label=variant)
            ax.fill_between(steps, q25, q75, color=colour, alpha=0.25, linewidth=0)

        ax.set_title(_pretty_name(metric) if title is None else title)
        if show_axes:
            ax.set_ylabel(_pretty_name(metric))
            ax.set_xlabel("Training steps (in thousands)")
            ax.xaxis.set_major_formatter(_thousands_formatter())
        else:
            ax.set_ylabel("")
            ax.set_xlabel("")
        ax.tick_params(axis="both", which="both", top=True, right=True)

    if show_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, 0.02), ncol=min(len(handles), 6),
                frameon=True, edgecolor="white", fancybox=False,
                columnspacing=1.5, handlelength=2.0,
            )

    fig.tight_layout(rect=[0, 0.08, 1, 1] if show_legend else [0, 0, 1, 1])

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)

    return fig


_PRETTY: Dict[str, str] = {
    "ep_reward": "Reward",
    "ep_length": "Episode Length",
    "cum_cost": "Cumulative Cost",
    "satisfied": "Safety Prob.",
    "override_rate": "Override Rate",
    "policy_loss": "Policy Loss",
    "value_loss": "Value Loss",
}


def _pretty_name(metric: str) -> str:
    """Map a metric key to a human-readable label."""
    # Strip common prefixes like "train/rollout/" or "eval/rollout/".
    short = metric.rsplit("/", 1)[-1] if "/" in metric else metric
    return _PRETTY.get(short, short.replace("_", " ").title())
