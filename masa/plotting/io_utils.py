"""Filesystem helpers and logging setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

LOGGER_NAME = "masa.plotting.probshield_plots"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                           datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_csv(path: Path, df: pd.DataFrame, **to_csv_kwargs) -> None:
    """Write a DataFrame via .tmp + os.replace so readers never see a half-file."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, **to_csv_kwargs)
    os.replace(tmp, path)
