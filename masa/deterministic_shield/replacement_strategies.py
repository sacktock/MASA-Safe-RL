"""Action replacement rules; every result must still pass the shield's safe mask."""
from __future__ import annotations

from collections.abc import Callable
import numpy as np

# Arguments: product observation, proposed action, copy of the safe-action mask.
Replacement = Callable[[int, int, np.ndarray], int]


def _candidates(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 1:
        raise ValueError("Expected a one-dimensional safe-action mask.")
    actions = np.flatnonzero(mask)
    if not actions.size:
        raise ValueError("Cannot select an action from an empty safe set.")
    return actions


def random_safe(seed: int | None = None) -> Replacement:
    """Sample uniformly from the current safe actions when replacement is needed.

    One private RNG is created per selector, not per call. The same seed and
    sequence of masks reproduce the same replacements. Use a separate selector
    for each environment to avoid sharing a random stream between environments.

    The RNG is independent of the environment's RNG: env.reset(seed=...) neither
    seeds nor rewinds this selector. Recreate random_safe(seed) to restart it.
    Safe proposals do not call the selector and therefore do not advance its RNG.
    """
    rng = np.random.default_rng(seed)

    def select(state: int, proposed: int, mask: np.ndarray) -> int:
        return int(rng.choice(_candidates(mask)))

    return select


def highest_score(score_fn: Callable[[int], np.ndarray]) -> Replacement:
    """Maximize score_fn(product_observation)[action] over safe actions.

    Use Q-values, policy logits or a priority vector. Ties choose the lowest
    action index. The rule is deterministic when score_fn is deterministic.
    It is called only when intervention is needed. Its result must have one
    entry per action and finite values for safe actions; unsafe entries are ignored.
    """
    if not callable(score_fn):
        raise TypeError("score_fn must be callable.")

    def select(state: int, proposed: int, mask: np.ndarray) -> int:
        actions = _candidates(mask)
        scores = np.asarray(score_fn(state), dtype=np.float64)
        if scores.shape != mask.shape:
            raise ValueError("score_fn must return one score per action.")
        if not np.isfinite(scores[actions]).all():
            raise ValueError("Scores of safe actions must be finite.")
        return int(actions[np.argmax(scores[actions])])

    return select
