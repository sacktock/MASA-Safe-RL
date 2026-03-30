from __future__ import annotations

from typing import Iterable

import numpy as np


def flatten_binary_obs(obs) -> np.ndarray:
    """Return a matrix-game observation as a flat 1D binary vector."""
    return np.asarray(obs, dtype=np.uint8).reshape(-1)


def binary_cost(labels: Iterable[str]) -> float:
    """Default multi-agent matrix-game cost: 1 iff the label set is unsafe."""
    return 1.0 if "unsafe" in labels else 0.0

def decode_lsb_bits(bits: np.ndarray) -> int:
    """Decode a little-endian binary vector into an integer."""
    value = 0
    for idx, bit in enumerate(np.asarray(bits, dtype=np.uint8).reshape(-1)):
        value |= (int(bit) & 1) << idx
    return value