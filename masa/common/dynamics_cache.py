"""On-disk cache for expensive, deterministic environment dynamics.

Some environments enumerate their transition dynamics at import time. For the
19x15 Pacman map that takes roughly two minutes, which is paid again by every
new process. The computation depends only on the layout and a handful of
scalars, so the result is cached on disk and keyed by those inputs.

The cache is best effort: any read or write failure falls back to recomputing,
so a missing, unreadable or stale cache slows startup but never breaks it.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar
import hashlib
import os
import pickle
import tempfile

import numpy as np


T = TypeVar("T")

# Bump when the cached payload's meaning changes, so stale files are ignored.
CACHE_FORMAT_VERSION = 1
_DISABLE_ENV_VAR = "MASA_DISABLE_DYNAMICS_CACHE"
_CACHE_DIR_ENV_VAR = "MASA_CACHE_DIR"


def cache_enabled() -> bool:
    """Whether caching is switched on (``MASA_DISABLE_DYNAMICS_CACHE`` opts out)."""
    return os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() not in ("1", "true", "yes")


def _repo_root() -> Path | None:
    """The checkout this package lives in, or ``None`` when installed elsewhere."""
    root = Path(__file__).resolve().parents[2]
    markers = (root / "pyproject.toml", root / ".git", root / "setup.py")
    return root if any(marker.exists() for marker in markers) else None


def cache_dir() -> Path:
    """Directory holding cached dynamics, honouring ``MASA_CACHE_DIR``.

    Defaults to ``.cache/dynamics`` in the checkout (``.cache/`` is gitignored),
    so the cache lives beside the code that produced it. When the package is
    installed outside a checkout there is nowhere sensible to write inside it,
    so it falls back to the user cache directory.
    """
    override = os.environ.get(_CACHE_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    root = _repo_root()
    if root is not None:
        return root / ".cache" / "dynamics"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "masa" / "dynamics"


def cache_key(name: str, *parts: Any) -> str:
    """Build a stable filename stem from ``name`` and the inputs in ``parts``."""
    digest = hashlib.sha256()
    digest.update(f"v{CACHE_FORMAT_VERSION}\0{name}".encode())
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(f"\0ndarray:{part.dtype}:{part.shape}\0".encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(f"\0{type(part).__name__}:{part!r}".encode())
    return f"{name}-{digest.hexdigest()[:16]}"


def cache_path(key: str) -> Path:
    return cache_dir() / f"{key}.pkl"


def _write_atomically(path: Path, value: Any) -> None:
    """Write via a temp file in the same directory, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)
    except BaseException:
        with suppress(OSError):
            temp_path.unlink()
        raise


def cached_dynamics(key: str, compute: Callable[[], T]) -> T:
    """Return ``compute()``, reading from and populating the on-disk cache.

    Cache problems are never fatal: an unreadable or unwritable cache just means
    the value is recomputed (and, where possible, rewritten).
    """
    if not cache_enabled():
        return compute()

    path = cache_path(key)
    if path.exists():
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            # Corrupt, truncated or written by an incompatible version.
            with suppress(OSError):
                path.unlink()

    value = compute()
    try:
        _write_atomically(path, value)
    except Exception:
        pass
    return value


__all__ = [
    "CACHE_FORMAT_VERSION",
    "cache_dir",
    "cache_enabled",
    "cache_key",
    "cache_path",
    "cached_dynamics",
]
