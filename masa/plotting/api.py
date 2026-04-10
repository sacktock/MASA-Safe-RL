"""MASA plotting utilities -- unified interface for TensorBoard & W&B."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from masa.plotting.plot import plot_metrics
from masa.plotting.sources import TensorBoardSource, WandBSource


def plot_run(
    *,
    # Source selection -- provide one of these pairs
    tensorboard_logdir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_filters: Optional[dict] = None,
    run_filter: Optional[str] = None,
    # What to plot
    metrics: Optional[List[str]] = None,
    # Plotting options
    smooth_weight: float = 0.6,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (11, 6),
    show_legend: bool = True,
    show_axes: bool = True,
    dpi: int = 300,
    palette: Optional[Dict[str, str]] = None,
    errorbar: Tuple[str, float] = ("sd", 1),
) -> plt.Figure:
    """Plot training metrics from TensorBoard or W&B.

    Provide **either** ``tensorboard_logdir`` or ``wandb_project`` to select
    the data source.  Metrics are discovered dynamically from the source;
    pass *metrics* to restrict which ones are plotted.

    Args:
        tensorboard_logdir: Path to TensorBoard log directory.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (optional, uses default if ``None``).
        wandb_filters: MongoDB-style filters passed to ``wandb.Api().runs()``.
        run_filter: Simple substring match on run / directory name.
        metrics: Metric keys to plot.  ``None`` plots all available.
        smooth_weight: EMA smoothing factor in ``[0, 1)``.
        title: Figure title.
        output_path: Save path (creates parent dirs).  ``None`` skips saving.
        figsize: ``(width, height)`` per subplot panel.
        show_legend: Render a legend above the panels.
        show_axes: Label axes with metric names and "Step".
        dpi: Figure DPI for saved files.
        palette: Colour mapping ``{run_name: colour}``.
        errorbar: Seaborn errorbar specification for shaded variance bands
            (default ``("sd", 1)`` for +/- 1 standard deviation).

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.

    Raises:
        ValueError: If neither or both sources are specified.
    """
    if tensorboard_logdir and wandb_project:
        raise ValueError("Specify tensorboard_logdir OR wandb_project, not both.")
    if not tensorboard_logdir and not wandb_project:
        raise ValueError("Provide either tensorboard_logdir or wandb_project.")

    if tensorboard_logdir:
        source = TensorBoardSource(tensorboard_logdir)
        df = source.load(metrics=metrics, run_filter=run_filter)
    else:
        source = WandBSource(wandb_project, entity=wandb_entity)
        df = source.load(metrics=metrics, filters=wandb_filters, run_filter=run_filter)

    return plot_metrics(
        df,
        metrics=metrics,
        smooth_weight=smooth_weight,
        title=title,
        output_path=output_path,
        figsize=figsize,
        show_legend=show_legend,
        show_axes=show_axes,
        dpi=dpi,
        palette=palette,
        errorbar=errorbar,
    )
