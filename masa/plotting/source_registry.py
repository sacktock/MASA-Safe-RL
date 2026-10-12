"""Registry of benchmark data sources.

Each backend wraps the loaders in ``sources.py`` behind a uniform interface
(``download`` / ``build_frames`` / ``discover_envs``) so the pipeline looks a
source up by ``config.source_kind`` and never branches on the backend. To add a
backend: subclass ``BenchmarkSource``, decorate with ``@register_source``, and
teach ``config._resolve_source`` to read its ``source:`` block.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Config
from .io_utils import get_logger


class BenchmarkSource(ABC):
    """Uniform interface over a benchmark data backend (W&B, TensorBoard, ...)."""

    kind: str = ""  # set by @register_source

    def __init__(self, config: Config):
        self.config = config
        self.log = get_logger()

    @abstractmethod
    def download(self) -> list[Path]:
        """Materialise raw data locally if needed; return any cached paths.

        May be a no-op for backends whose data is already on disk.
        """

    @abstractmethod
    def build_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(long_form, quantile)`` frames in the canonical schema."""

    @abstractmethod
    def discover_envs(self) -> list[str]:
        """Best-effort env names for the dry-run preview (no heavy data load)."""

    def describe(self) -> str:
        """One-line human description for dry-run logging."""
        return self.kind


_REGISTRY: dict[str, type[BenchmarkSource]] = {}


def register_source(kind: str) -> Callable[[type[BenchmarkSource]], type[BenchmarkSource]]:
    """Class decorator registering a :class:`BenchmarkSource` under ``kind``."""

    def deco(cls: type[BenchmarkSource]) -> type[BenchmarkSource]:
        if kind in _REGISTRY:
            raise ValueError(f"source kind collision: {kind!r} already registered.")
        cls.kind = kind
        _REGISTRY[kind] = cls
        return cls

    return deco


def get_source(config: Config) -> BenchmarkSource:
    """Instantiate the source registered for ``config.source_kind``."""
    kind = config.source_kind
    if kind not in _REGISTRY:
        raise KeyError(
            f"unknown source kind {kind!r}; registered: {registered_kinds()}"
        )
    return _REGISTRY[kind](config)


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)


@register_source("wandb")
class WandbBenchmarkSource(BenchmarkSource):
    """W&B backend: download per-run CSVs, then build frames from that cache."""

    def download(self) -> list[Path]:
        from .sources import CachedWandbSource

        return CachedWandbSource(self.config).download()

    def build_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        from .processing import build_frames

        return build_frames(self.config)

    def discover_envs(self) -> list[str]:
        cfg = self.config
        envs: list[str] = []
        if cfg.cache_dir.exists():
            pat = cfg.run_schema.compile()
            for p in sorted(cfg.cache_dir.glob("*.csv")):
                m = pat.match(p.stem)
                if m:
                    envs.append(m.group("env"))
        return sorted(set(envs))

    def describe(self) -> str:
        c = self.config
        return f"wandb ({c.wandb_entity}/{c.wandb_project})"


@register_source("tensorboard")
class TensorBoardBenchmarkSource(BenchmarkSource):
    """TensorBoard backend: read local event files in place (no download)."""

    def download(self) -> list[Path]:
        self.log.info("tensorboard source: reading local event files, no download step.")
        return []

    def build_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        from .processing import build_frames_from_tb

        return build_frames_from_tb(self.config)

    def discover_envs(self) -> list[str]:
        from .processing import tb_run_dirs_by_env

        return sorted(tb_run_dirs_by_env(self.config.tensorboard_logdir))

    def describe(self) -> str:
        return f"tensorboard ({self.config.tensorboard_logdir})"
