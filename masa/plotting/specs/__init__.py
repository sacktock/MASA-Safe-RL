"""PlotSpec registry.

Importing this module registers the generic ``performance`` spec. Extra spec
modules (for example ``masa.plotting.examples.probshield``) can be loaded at
runtime via the ``--specs-from <module>`` CLI flag or by importing them from
Python before resolving specs.
"""

from __future__ import annotations

from typing import Optional

from .base import Band, HLine, Panel, PanelSource, PlotSpec, RenderContext

_REGISTRY: dict[str, PlotSpec] = {}


def register(spec: PlotSpec) -> PlotSpec:
    if spec.id in _REGISTRY:
        raise ValueError(f"PlotSpec id collision: {spec.id!r} already registered.")
    _REGISTRY[spec.id] = spec
    return spec


def get(spec_id: str) -> PlotSpec:
    if spec_id not in _REGISTRY:
        raise KeyError(f"Unknown PlotSpec id: {spec_id!r}. "
                       f"Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[spec_id]


def all_specs(filter_ids: Optional[list[str]] = None) -> list[PlotSpec]:
    if filter_ids is None:
        return list(_REGISTRY.values())
    return [get(i) for i in filter_ids]


# Auto register the generic spec that ships with the library.
from . import performance  # noqa: E402, F401
from . import margin       # noqa: E402, F401


__all__ = [
    "Band", "HLine", "Panel", "PanelSource", "PlotSpec", "RenderContext",
    "register", "get", "all_specs",
]
