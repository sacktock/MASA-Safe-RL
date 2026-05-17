"""Read per-run CSVs and produce the canonical long-form + quantile frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .io_utils import atomic_write_csv, get_logger

LONG_FILENAME = "long.csv"
QUANTILES_FILENAME = "quantiles.csv"
_CACHE_INTERNAL = {LONG_FILENAME, QUANTILES_FILENAME, "grouped_results.csv"}


def build_long_form(config: Config) -> pd.DataFrame:
    """Melt every per-run CSV in ``cache_dir`` into [env, variant, seed, step, metric, value]."""
    log = get_logger()
    pattern = config.run_schema.compile()
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(config.cache_dir.glob("*.csv")):
        if csv_path.name in _CACHE_INTERNAL:
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


def load_or_build(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_path = config.cache_dir / LONG_FILENAME
    quantile_path = config.cache_dir / QUANTILES_FILENAME
    log = get_logger()

    if (not config.force_process
            and long_path.exists()
            and quantile_path.exists()):
        log.info(f"Using cached processed frames in {config.cache_dir}")
        return pd.read_csv(long_path), pd.read_csv(quantile_path)

    log.info("Building long-form frame from per-run CSVs ...")
    long_df = build_long_form(config)
    log.info(f"  long-form: {len(long_df)} rows, "
             f"{long_df['env'].nunique() if not long_df.empty else 0} envs, "
             f"{long_df['variant'].nunique() if not long_df.empty else 0} variants")

    log.info("Building quantile frame ...")
    q_df = build_quantile_frame(long_df)
    log.info(f"  quantile frame: {len(q_df)} rows")

    atomic_write_csv(long_path, long_df)
    atomic_write_csv(quantile_path, q_df)
    return long_df, q_df
