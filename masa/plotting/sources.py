"""Data sources for MASA plotting.

TensorBoardSource and WandBSource are long-form loaders for the single-run
plotter (returning ``[step, metric, value, run]``). CachedWandbSource is the
benchmark-pipeline downloader, caching each W&B run's history to a per-run CSV.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


class TensorBoardSource:
    """Read scalar metrics from TensorBoard event files.

    Args:
        logdir: Root directory containing per-run subdirectories
            (``{logdir}/{run_name}/events.out.tfevents.*``). If *logdir* itself
            contains event files it is treated as a single run.
    """

    def __init__(self, logdir: str):
        self.logdir = logdir

    def list_metrics(self) -> List[str]:
        """Return all tensor/scalar tag names found across runs."""
        from tensorboard.backend.event_processing import event_accumulator

        tags: set[str] = set()
        for run_dir in self._find_run_dirs():
            ea = event_accumulator.EventAccumulator(
                run_dir,
                size_guidance={event_accumulator.SCALARS: 0,
                               event_accumulator.TENSORS: 0},
            )
            ea.Reload()
            tags.update(ea.Tags().get("tensors", []))
            tags.update(ea.Tags().get("scalars", []))
        return sorted(tags)

    def load(
        self,
        metrics: Optional[List[str]] = None,
        run_filter: Optional[str | List[str]] = None,
    ) -> pd.DataFrame:
        """Load metrics into a ``[step, metric, value, run]`` DataFrame."""
        from tensorboard.backend.event_processing import event_accumulator
        from tensorboard.util import tensor_util

        filters = [run_filter] if isinstance(run_filter, str) else (run_filter or [])
        frames: list[pd.DataFrame] = []

        for run_dir in self._find_run_dirs():
            run_name = os.path.basename(run_dir)
            if filters and not any(f in run_name for f in filters):
                continue

            ea = event_accumulator.EventAccumulator(
                run_dir,
                size_guidance={event_accumulator.SCALARS: 0,
                               event_accumulator.TENSORS: 0},
            )
            ea.Reload()

            available_tensors = set(ea.Tags().get("tensors", []))
            available_scalars = set(ea.Tags().get("scalars", []))
            targets = metrics if metrics else sorted(available_tensors | available_scalars)

            for tag in targets:
                if tag in available_tensors:
                    rows = []
                    for e in ea.Tensors(tag):
                        arr = tensor_util.make_ndarray(e.tensor_proto)
                        if arr.size != 1:
                            # Skip non-scalar tensors (e.g. tf.summary.histogram
                            # margins) -- the scalar plotter can't use them.
                            continue
                        rows.append({"step": e.step, "value": arr.item(),
                                     "metric": tag, "run": run_name})
                elif tag in available_scalars:
                    rows = [
                        {"step": e.step, "value": e.value,
                         "metric": tag, "run": run_name}
                        for e in ea.Scalars(tag)
                    ]
                else:
                    continue

                if rows:
                    frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(columns=["step", "metric", "value", "run"])
        return pd.concat(frames, ignore_index=True)

    def _find_run_dirs(self) -> List[str]:
        """Discover directories that contain TensorBoard event files."""
        run_dirs: list[str] = []
        for root, _dirs, files in os.walk(self.logdir):
            if any(f.startswith("events.out.tfevents") for f in files):
                run_dirs.append(root)
        if not run_dirs:
            raise FileNotFoundError(f"No TensorBoard event files found under {self.logdir}")
        return sorted(run_dirs)


class WandBSource:
    """Read scalar metrics from Weights & Biases.

    Args:
        project: W&B project name.
        entity: W&B entity / team. ``None`` uses the default entity.
    """

    def __init__(self, project: str, entity: Optional[str] = None):
        self.project = project
        self.entity = entity

    def list_metrics(self, filters: Optional[dict] = None) -> List[str]:
        """Return the union of history column names across matching runs."""
        import wandb

        api = wandb.Api()
        runs = api.runs(self._path(), filters=filters or {})
        cols: set[str] = set()
        for run in runs:
            hist = run.history(samples=1)
            cols.update(c for c in hist.columns if not c.startswith("_"))
        return sorted(cols)

    def load(
        self,
        metrics: Optional[List[str]] = None,
        filters: Optional[dict] = None,
        run_filter: Optional[str | List[str]] = None,
    ) -> pd.DataFrame:
        """Load metrics into a ``[step, metric, value, run]`` DataFrame.

        Args:
            metrics: Metric keys to load; ``None`` loads all non-internal columns.
            filters: W&B MongoDB-style run filters (passed to ``api.runs``).
            run_filter: Substring(s) matched against ``run.name``.
        """
        import wandb

        api = wandb.Api()
        runs = api.runs(self._path(), filters=filters or {})

        run_filters = [run_filter] if isinstance(run_filter, str) else (run_filter or [])
        frames: list[pd.DataFrame] = []

        for run in runs:
            if run_filters and not any(f in run.name for f in run_filters):
                continue

            hist = run.history(samples=10_000)
            if hist.empty:
                continue

            available = [c for c in hist.columns if not c.startswith("_")]
            targets = metrics if metrics else available

            for col in targets:
                if col not in hist.columns:
                    continue
                sub = hist[["_step", col]].dropna(subset=[col])
                if sub.empty:
                    continue
                sub = sub.rename(columns={"_step": "step", col: "value"})
                sub["metric"] = col
                sub["run"] = run.name
                frames.append(sub[["step", "metric", "value", "run"]])

        if not frames:
            return pd.DataFrame(columns=["step", "metric", "value", "run"])
        return pd.concat(frames, ignore_index=True)

    def _path(self) -> str:
        return f"{self.entity}/{self.project}" if self.entity else self.project


from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config
from .io_utils import atomic_write_csv, ensure_dir, get_logger


_RETRY = dict(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)


class CachedWandbSource:
    """Per-run CSV downloader with bounded retries.

    Writes each run's history to ``Config.cache_dir/{run.name}.csv``, skipping
    runs already cached unless ``Config.force_download`` is set. Per-run failures
    are logged and skipped, never raised.
    """

    def __init__(self, config: Config):
        self.config = config
        self.log = get_logger()

    def download(self) -> list[Path]:
        """Download (or read from cache) every run; return the per-run CSV paths."""
        ensure_dir(self.config.cache_dir)
        try:
            runs = list(self._list_runs())
        except Exception as e:
            self.log.error(f"W&B run listing failed after retries: {e}")
            return self._existing_csvs()

        self.log.info(f"W&B reports {len(runs)} runs in {self._project_path()}")
        paths: list[Path] = []
        for run in runs:
            csv_path = self.config.cache_dir / f"{run.name}.csv"
            if csv_path.exists() and not self.config.force_download:
                self.log.debug(f"  cache hit: {run.name}")
                paths.append(csv_path)
                continue
            try:
                df = self._fetch_history(run)
            except Exception as e:
                self.log.warning(f"  skipping {run.name}: {e}")
                continue
            atomic_write_csv(csv_path, df)
            self.log.info(f"  downloaded: {run.name}  ({len(df)} rows)")
            paths.append(csv_path)
        return paths

    def _project_path(self) -> str:
        c = self.config
        return f"{c.wandb_entity}/{c.wandb_project}" if c.wandb_entity else c.wandb_project

    @retry(**_RETRY)
    def _list_runs(self) -> Iterable:
        import wandb
        api = wandb.Api()
        return api.runs(self._project_path())

    @retry(**_RETRY)
    def _fetch_history(self, run) -> pd.DataFrame:
        df = run.history(samples=10_000)
        if df.empty:
            raise RuntimeError("history is empty")
        return df

    def _existing_csvs(self) -> list[Path]:
        if not self.config.cache_dir.exists():
            return []
        return sorted(p for p in self.config.cache_dir.glob("*.csv")
                      if p.name != "grouped_results.csv")
