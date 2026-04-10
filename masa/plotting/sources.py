"""Data sources that normalize TensorBoard and W&B logs into a common DataFrame."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd


class TensorBoardSource:
    """Read scalar metrics from TensorBoard event files.

    Args:
        logdir: Root directory containing per-run subdirectories.
            Expected layout: ``{logdir}/{run_name}/events.out.tfevents.*``
            If *logdir* itself contains event files it is treated as a single run.
    """

    def __init__(self, logdir: str):
        self.logdir = logdir

    # ------------------------------------------------------------------
    def list_metrics(self) -> List[str]:
        """Return all tensor tag names found across runs."""
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

    # ------------------------------------------------------------------
    def load(
        self,
        metrics: Optional[List[str]] = None,
        run_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load metrics into a normalised DataFrame.

        Returns:
            DataFrame with columns ``[step, metric, value, run]``.
        """
        import tensorflow as tf
        from tensorboard.backend.event_processing import event_accumulator

        frames: list[pd.DataFrame] = []

        for run_dir in self._find_run_dirs():
            run_name = os.path.basename(run_dir)
            if run_filter and run_filter not in run_name:
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
                    events = ea.Tensors(tag)
                    rows = [
                        {"step": e.step,
                         "value": tf.make_ndarray(e.tensor_proto).item(),
                         "metric": tag,
                         "run": run_name}
                        for e in events
                    ]
                elif tag in available_scalars:
                    events = ea.Scalars(tag)
                    rows = [
                        {"step": e.step,
                         "value": e.value,
                         "metric": tag,
                         "run": run_name}
                        for e in events
                    ]
                else:
                    continue

                if rows:
                    frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(columns=["step", "metric", "value", "run"])
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    def _find_run_dirs(self) -> List[str]:
        """Discover directories that contain TensorBoard event files."""
        run_dirs: list[str] = []
        for root, _dirs, files in os.walk(self.logdir):
            if any(f.startswith("events.out.tfevents") for f in files):
                run_dirs.append(root)
        if not run_dirs:
            raise FileNotFoundError(
                f"No TensorBoard event files found under {self.logdir}"
            )
        return sorted(run_dirs)


class WandBSource:
    """Read scalar metrics from Weights & Biases.

    Args:
        project: W&B project name (e.g. ``"ProbShield-Benchmarks"``).
        entity: W&B entity / team.  ``None`` uses the default entity.
    """

    def __init__(self, project: str, entity: Optional[str] = None):
        self.project = project
        self.entity = entity

    # ------------------------------------------------------------------
    def list_metrics(self, filters: Optional[dict] = None) -> List[str]:
        """Return the union of history column names across matching runs."""
        import wandb

        api = wandb.Api()
        runs = api.runs(
            self._path(),
            filters=filters or {},
        )
        cols: set[str] = set()
        for run in runs:
            hist = run.history(samples=1)
            cols.update(c for c in hist.columns if not c.startswith("_"))
        return sorted(cols)

    # ------------------------------------------------------------------
    def load(
        self,
        metrics: Optional[List[str]] = None,
        filters: Optional[dict] = None,
        run_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load metrics into a normalised DataFrame.

        Args:
            metrics: Metric keys to load.  If ``None``, loads all non-internal
                columns from run history.
            filters: W&B MongoDB-style run filters (passed to ``api.runs``).
            run_filter: Simple substring filter on ``run.name``.

        Returns:
            DataFrame with columns ``[step, metric, value, run]``.
        """
        import wandb

        api = wandb.Api()
        runs = api.runs(
            self._path(),
            filters=filters or {},
        )

        frames: list[pd.DataFrame] = []

        for run in runs:
            if run_filter and run_filter not in run.name:
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

    # ------------------------------------------------------------------
    def _path(self) -> str:
        if self.entity:
            return f"{self.entity}/{self.project}"
        return self.project