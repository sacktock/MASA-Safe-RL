"""Orchestrates download -> process -> render with per-stage error boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .config import Config
from .io_utils import get_logger
from .processing import load_or_build
from .sources import CachedWandbSource
from .specs import PlotSpec, all_specs
from .specs.base import RenderContext

VALID_STAGES = ("download", "process", "render")


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.log = get_logger()

    # -- individual stages ------------------------------------------------

    def download(self) -> list[Path]:
        try:
            source = CachedWandbSource(self.config)
            return source.download()
        except Exception as e:
            self.log.error(f"download stage crashed: {e}")
            return []

    def process(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return load_or_build(self.config)

    def render(self,
               long_df: pd.DataFrame,
               quantile_df: pd.DataFrame,
               specs: Sequence[PlotSpec]) -> list[Path]:
        from .render import apply_rc, render_spec

        apply_rc()
        if long_df.empty:
            self.log.error("render stage aborted: long-form frame is empty.")
            return []
        envs = sorted(long_df["env"].unique())
        self.log.info(f"Rendering {len(specs)} spec(s) across {len(envs)} env(s).")
        outputs: list[Path] = []
        for spec in specs:
            self.log.info(f"Spec '{spec.id}' -> {self.config.output_dir / spec.out_subdir}")
            targets = envs if spec.per_env else ["__all__"]
            for env in targets:
                ctx = self._make_ctx(env, long_df, quantile_df)
                try:
                    path = render_spec(spec, ctx, self.config.output_dir)
                except Exception as e:
                    self.log.warning(f"  spec '{spec.id}' / env '{env}' crashed: {e}")
                    continue
                if path is not None:
                    outputs.append(path)
        return outputs

    # -- orchestration ----------------------------------------------------

    def run(self,
            specs: Optional[Sequence[PlotSpec]] = None,
            stages: Sequence[str] = VALID_STAGES,
            dry_run: bool = False) -> list[Path]:
        if dry_run:
            self._dry_run(specs)
            return []

        long_df: Optional[pd.DataFrame] = None
        q_df: Optional[pd.DataFrame] = None

        if "download" in stages:
            paths = self.download()
            self.log.info(f"Download stage: {len(paths)} per-run CSVs available.")

        if "process" in stages:
            try:
                long_df, q_df = self.process()
            except Exception as e:
                self.log.error(f"process stage failed: {e}")
                return []

        if "render" in stages:
            if long_df is None or q_df is None:
                try:
                    long_df, q_df = self.process()
                except Exception as e:
                    self.log.error(f"render needs processed data; process failed: {e}")
                    return []
            chosen = specs if specs is not None else all_specs()
            return self.render(long_df, q_df, chosen)

        return []

    # -- helpers ----------------------------------------------------------

    def _make_ctx(self, env: str, long_df: pd.DataFrame, q_df: pd.DataFrame) -> RenderContext:
        env_metrics = (long_df.loc[long_df["env"] == env, "metric"].unique()
                       if not long_df.empty else [])
        return RenderContext(
            env=env,
            long_df=long_df,
            quantile_df=q_df,
            variants=list(self.config.variants),
            seed_for_logged_quantiles=self.config.seed_for_logged_quantiles,
            available_metrics=frozenset(env_metrics),
        )

    def _dry_run(self, specs: Optional[Sequence[PlotSpec]]) -> None:
        cfg = self.config
        self.log.info("=== DRY RUN ===")
        self.log.info(f"W&B          : {cfg.wandb_entity}/{cfg.wandb_project}")
        self.log.info(f"cache_dir    : {cfg.cache_dir}")
        self.log.info(f"output_dir   : {cfg.output_dir}")
        self.log.info(f"variants     : {[v.name for v in cfg.variants]}")
        self.log.info(f"palette      : {cfg.palette}")
        chosen = specs if specs is not None else all_specs()
        self.log.info(f"specs        : {[s.id for s in chosen]}")
        envs: list[str] = []
        if cfg.cache_dir.exists():
            pat = cfg.run_schema.compile()
            for p in sorted(cfg.cache_dir.glob("*.csv")):
                m = pat.match(p.stem)
                if m:
                    envs.append(m.group("env"))
            envs = sorted(set(envs))
        self.log.info(f"envs (cache) : {envs or '(none discovered yet)'}")
        for spec in chosen:
            for env in envs or ["<env>"]:
                target = cfg.output_dir / spec.out_subdir / f"{spec.id}_{env}.png"
                self.log.info(f"  would write {target}")
