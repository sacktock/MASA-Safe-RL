"""Read per-run CSVs and produce the canonical long-form + quantile frames.

The aggregated frames are rebuilt in memory on every invocation. The expensive
step (downloading run histories from W&B) is still cached per-run by
``CachedWandbSource``; the melt + groupby here is fast over local CSVs and not
worth the staleness risk of a persistent cache.

Some columns (notably ``train/stats/margin_<t>``) ship as stringified
``wandb.Histogram`` payloads rather than scalars. ``_coerce_value`` collapses
each histogram to its (linearly interpolated) median so it can flow through the
same long-form / quantile machinery as any other scalar metric.
"""

from __future__ import annotations

import ast
import math

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
    return pd.concat(frames, ignore_index=True)


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
