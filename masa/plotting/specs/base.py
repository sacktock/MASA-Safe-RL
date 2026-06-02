"""Dataclasses describing figures declaratively. Consumed by render.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..config import VariantStyle

PanelSource = Literal["aggregated", "per_seed", "logged_quantile"]


@dataclass(frozen=True)
class Band:
    """Shaded band between two quantile column names (or _qXX suffixes)."""
    lower: str
    upper: str
    alpha: float = 0.3


@dataclass(frozen=True)
class HLine:
    y: float
    style: str = ":"
    colour: str = "gray"
    alpha: float = 0.5
    linewidth: float = 0.8


@dataclass(frozen=True)
class Panel:
    """One subplot within a figure.

    For ``source="aggregated"``:
        ``metric`` matches a value in the quantile frame; ``central`` selects
        the line column (e.g. ``q50``); each ``Band`` references two columns.

    For ``source="per_seed"``:
        ``metric`` selects long-form rows; one line per seed is drawn,
        seed 0 / first solid, others dashed with reduced alpha.

    For ``source="logged_quantile"``:
        ``metric`` is the *base* metric name; bands reference suffixes like
        ``q05``/``q95``; only ``seed_for_logged_quantiles`` is read.
    """
    metric: str
    title: str
    source: PanelSource = "aggregated"
    ylabel: Optional[str] = None
    xlabel: Optional[str] = None
    smooth_alpha: Optional[float] = None
    central: str = "q50"
    bands: tuple[Band, ...] = ()
    hlines: tuple[HLine, ...] = ()
    ylim: Optional[tuple[float, float]] = None
    variants_filter: Optional[tuple[str, ...]] = None
    custom: Optional[Callable[["Axes", "RenderContext"], None]] = None


PanelsFactory = Callable[["RenderContext"], Sequence[Panel]]
GridFactory = Callable[[Sequence[Panel]], tuple[int, int]]


FigureBuilder = Callable[["Figure", "RenderContext", set], None]


@dataclass(frozen=True)
class PlotSpec:
    """Declarative description of one figure type.

    Two modes:
      - panel mode (default): ``panels`` + ``grid`` drive a uniform plt.subplots
        layout; the renderer dispatches each panel by ``Panel.source``.
      - builder mode: when ``figure_builder`` is set, the renderer creates an
        empty figure and hands it to the builder, which is free to use GridSpec
        or any layout. Builder mutates the ``drawn`` set to register which
        variants appear for legend purposes.
    """
    id: str
    panels: Union[tuple[Panel, ...], PanelsFactory] = ()
    grid: Union[tuple[int, int], GridFactory] = (1, 1)
    suptitle_fmt: str = "{env_title}"
    figsize: Optional[tuple[float, float]] = None
    out_subdir: str = ""
    per_env: bool = True
    show_legend: bool = True
    figure_builder: Optional[FigureBuilder] = None

    def resolve_panels(self, ctx: "RenderContext") -> tuple[Panel, ...]:
        if callable(self.panels):
            return tuple(self.panels(ctx))
        return tuple(self.panels)

    def resolve_grid(self, panels: Sequence[Panel]) -> tuple[int, int]:
        if callable(self.grid):
            return self.grid(panels)
        return self.grid


@dataclass(frozen=True)
class RenderContext:
    """Per-figure data passed to panel factories and custom panels."""
    env: str
    long_df: "pd.DataFrame"          # full long-form (all envs)
    quantile_df: "pd.DataFrame"      # full quantile frame (all envs)
    variants: list["VariantStyle"]   # ordered, with colours filled
    seed_for_logged_quantiles: int
    available_metrics: frozenset[str] = field(default_factory=frozenset)
