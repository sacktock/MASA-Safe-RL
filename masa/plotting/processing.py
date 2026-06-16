"""Build the canonical long-form and quantile frames from cached runs.

Frames are rebuilt in memory each run (only the W&B download is cached on disk).
Stringified ``wandb.Histogram`` cells are collapsed to their median by
``_coerce_value`` so they flow through the same machinery as scalar metrics.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config
from .io_utils import get_logger

# Legacy artefacts from the old persistent-cache layout. Skipped silently so
# they don't trigger "does not match run_schema" warnings. Safe to delete from
# disk; kept here in case old caches are still around.
_LEGACY_NON_RUN_FILES = {"long.csv", "quantiles.csv", "grouped_results.csv"}


def _histogram_median(payload: dict) -> float:
    """Linearly-interpolated median of a ``wandb.Histogram`` dict payload.

    Accepts either explicit ``bins`` (N+1 edges) or the compact ``packedBins``
    {count, min, size} form. Returns NaN for malformed or empty histograms.
    """
    values = payload.get("values")
    if not values:
        return float("nan")
    if "bins" in payload:
        bins = payload["bins"]
    elif "packedBins" in payload:
        pb = payload["packedBins"]
        n, mn, sz = pb.get("count"), pb.get("min"), pb.get("size")
        if n is None or mn is None or sz is None:
            return float("nan")
        bins = [mn + i * sz for i in range(n + 1)]
    else:
        return float("nan")
    if len(bins) != len(values) + 1:
        return float("nan")
    total = float(sum(values))
    if total <= 0:
        return float("nan")
    half = total / 2.0
    cum = 0.0
    for i, count in enumerate(values):
        nxt = cum + count
        if nxt >= half and count > 0:
            frac = (half - cum) / count
            return bins[i] + frac * (bins[i + 1] - bins[i])
        cum = nxt
    return float(bins[-1])


def _coerce_value(v):
    """Convert a melted cell to float; NaN if not a scalar or parseable histogram."""
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        f = float(v)
        return f if not math.isnan(f) else float("nan")
    s = str(v).strip()
    if not s:
        return float("nan")
    if s[0] == "{":
        try:
            payload = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return float("nan")
        if isinstance(payload, dict) and payload.get("_type") == "histogram":
            return _histogram_median(payload)
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def build_long_form(config: Config) -> pd.DataFrame:
    """Melt every per-run CSV in ``cache_dir`` into [env, variant, seed, step, metric, value]."""
    log = get_logger()
    pattern = config.run_schema.compile()
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(config.cache_dir.glob("*.csv")):
        if csv_path.name in _LEGACY_NON_RUN_FILES:
            continue
        run_name = csv_path.stem
        match = pattern.match(run_name)
        if not match:
            log.warning(f"  skipping {run_name}: does not match run_schema")
            continue
        env = match.group("env")
        variant = match.group("variant")
        try:
            seed = int(match.group("seed"))
        except (TypeError, ValueError):
            log.warning(f"  skipping {run_name}: non-integer seed")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            log.warning(f"  skipping {run_name}: read failed ({e})")
            continue
        if "_step" not in df.columns:
            log.warning(f"  skipping {run_name}: no _step column")
            continue
        value_cols = [c for c in df.columns if c != "_step" and not c.startswith("_")]
        if not value_cols:
            log.warning(f"  skipping {run_name}: no metric columns")
            continue
        long = (
            df[["_step"] + value_cols]
            .melt(id_vars=["_step"], var_name="metric", value_name="value")
            .assign(value=lambda d: d["value"].map(_coerce_value))
            .dropna(subset=["value"])
            .rename(columns={"_step": "step"})
        )
        long["env"] = env
        long["variant"] = variant
        long["seed"] = seed
        frames.append(long[["env", "variant", "seed", "step", "metric", "value"]])

    if not frames:
        return pd.DataFrame(columns=["env", "variant", "seed", "step", "metric", "value"])
    out = pd.concat(frames, ignore_index=True)
    if config.max_step is not None:
        out = out[out["step"] <= config.max_step].reset_index(drop=True)
    return out


def build_quantile_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds: [env, variant, step, metric, q05, q25, q50, q75, q95]."""
    cols = ["env", "variant", "step", "metric", "q05", "q25", "q50", "q75", "q95"]
    if long_df.empty:
        return pd.DataFrame(columns=cols)
    g = long_df.groupby(["env", "variant", "step", "metric"], sort=True)["value"]
    out = pd.DataFrame({
        "q05": g.quantile(0.05),
        "q25": g.quantile(0.25),
        "q50": g.quantile(0.50),
        "q75": g.quantile(0.75),
        "q95": g.quantile(0.95),
    }).reset_index()
    return out[cols]


def build_long_form_from_tb(
    raw_df: pd.DataFrame,
    env: str,
    pattern: str = r"^(?P<variant>.+)_(?P<seed>\d+)$",
) -> pd.DataFrame:
    """Parse a TensorBoardSource raw frame into [env, variant, seed, step, metric, value].

    Mirrors build_long_form but takes the in-memory DataFrame from
    TensorBoardSource.load instead of per-run CSVs.  The default pattern
    handles directory names like ``CPO_0`` or ``QL_LAMBDA_3``.
    """
    log = get_logger()
    pat = re.compile(pattern)
    frames: list[pd.DataFrame] = []
    for run_name, grp in raw_df.groupby("run"):
        m = pat.match(run_name)
        if not m:
            log.warning(f"  skipping {run_name}: does not match pattern")
            continue
        try:
            seed = int(m.group("seed"))
        except (TypeError, ValueError):
            log.warning(f"  skipping {run_name}: non-integer seed")
            continue
        sub = grp[["step", "metric", "value"]].copy()
        sub["env"] = env
        sub["variant"] = m.group("variant")
        sub["seed"] = seed
        frames.append(sub[["env", "variant", "seed", "step", "metric", "value"]])
    if not frames:
        return pd.DataFrame(columns=["env", "variant", "seed", "step", "metric", "value"])
    return pd.concat(frames, ignore_index=True)


def tb_run_dirs_by_env(logdir) -> dict[str, list[str]]:
    """Map ``env -> [leaf run directories]`` under a TensorBoard benchmark root.

    ``env`` is the run directory's parent name, or ``logdir`` itself when run
    directories sit directly under it (i.e. ``logdir`` is a single env). Walks
    the filesystem for event files only -- no EventAccumulator load -- so it is
    cheap enough for dry-run previews.
    """
    from .sources import TensorBoardSource

    root = Path(logdir)
    if not root.exists():
        return {}
    try:
        run_dirs = TensorBoardSource(str(root))._find_run_dirs()
    except FileNotFoundError:
        return {}
    out: dict[str, list[str]] = {}
    for run_dir in run_dirs:
        parent = Path(run_dir).parent
        env = root.name if parent == root else parent.name
        out.setdefault(env, []).append(run_dir)
    return out


def build_long_form_from_tb_root(
    logdir: Path,
    pattern: str = r"^(?P<variant>.+)_(?P<seed>\d+)$",
    max_step: Optional[int] = None,
) -> pd.DataFrame:
    """Build the canonical long form from a TensorBoard benchmark root.

    Expected layout (matches the training CLI's benchmark output)::

        {logdir}/{env}/{variant}_{seed}/events.out.tfevents.*

    Each leaf run directory is read with :class:`TensorBoardSource`; ``env`` is
    taken from the parent directory name (or ``logdir`` itself when run
    directories sit directly under it), and ``(variant, seed)`` are parsed from
    the leaf name via ``pattern``. The TensorBoard analogue of
    :func:`build_long_form`, which reads the W&B per-run CSV cache instead.
    """
    from .sources import TensorBoardSource

    log = get_logger()
    cols = ["env", "variant", "seed", "step", "metric", "value"]
    by_env = tb_run_dirs_by_env(logdir)
    if not by_env:
        log.error(f"no TensorBoard event files found under {logdir}")
        return pd.DataFrame(columns=cols)

    frames: list[pd.DataFrame] = []
    for env, run_dirs in by_env.items():
        for run_dir in run_dirs:
            raw = TensorBoardSource(run_dir).load()
            if raw.empty:
                continue
            long = build_long_form_from_tb(raw, env=env, pattern=pattern)
            if not long.empty:
                frames.append(long)

    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    if max_step is not None:
        out = out[out["step"] <= max_step].reset_index(drop=True)
    return out


def build_frames_from_tb(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form and quantile frames from a TensorBoard logdir.

    The TensorBoard analogue of :func:`build_frames`: instead of reading the
    W&B per-run CSV cache, it reads event files under ``config.tensorboard_logdir``.
    """
    log = get_logger()

    log.info("Building long-form frame from TensorBoard event files ...")
    long_df = build_long_form_from_tb_root(
        config.tensorboard_logdir,
        pattern=config.run_schema.pattern,
        max_step=config.max_step,
    )
    log.info(f"  long-form: {len(long_df)} rows, "
             f"{long_df['env'].nunique() if not long_df.empty else 0} envs, "
             f"{long_df['variant'].nunique() if not long_df.empty else 0} variants")

    log.info("Building quantile frame ...")
    q_df = build_quantile_frame(long_df)
    log.info(f"  quantile frame: {len(q_df)} rows")

    return long_df, q_df


def build_frames(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form and quantile frames in memory from the per-run CSVs."""
    log = get_logger()

    log.info("Building long-form frame from per-run CSVs ...")
    long_df = build_long_form(config)
    log.info(f"  long-form: {len(long_df)} rows, "
             f"{long_df['env'].nunique() if not long_df.empty else 0} envs, "
             f"{long_df['variant'].nunique() if not long_df.empty else 0} variants")

    log.info("Building quantile frame ...")
    q_df = build_quantile_frame(long_df)
    log.info(f"  quantile frame: {len(q_df)} rows")

    return long_df, q_df
