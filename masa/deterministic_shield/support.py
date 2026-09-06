from __future__ import annotations

import gymnasium as gym
import numpy as np


def read_support(
    base: gym.Env, n_states: int, n_actions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read MASA dynamics as successors[s, k] and support[s, a, k].

    Prefer the sparse dictionary. Dense MASA kernels use P[next, current, action].
    Missing/zero-mass actions are disabled, not vacuously safe.
    """
    if getattr(base, "has_successor_states_dict", False):
        successor_dict, probability_dict = base.get_successor_states_dict()
        transition = None
    elif getattr(base, "has_transition_matrix", False):
        transition = np.asarray(base.get_transition_matrix())
        if transition.shape != (n_states, n_states, n_actions):
            raise ValueError("Expected transition shape (next_state, state, action).")
    else:
        raise ValueError("The base environment must expose a finite transition model.")

    rows = []
    for s in range(n_states):
        if transition is None:
            ids = np.asarray(successor_dict.get(s, []))
            if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
                raise ValueError(f"Successors of state {s} must be integer IDs.")
            if np.any((ids < 0) | (ids >= n_states)):
                raise ValueError(f"Successor of state {s} is out of range.")
            ids = ids.astype(np.intp)
            zero = np.zeros(ids.size)
            p = np.asarray(
                [probability_dict.get((s, a), zero) for a in range(n_actions)],
                dtype=np.float64,
            )
        else:
            ids = np.arange(n_states, dtype=np.intp)
            p = np.asarray(transition[:, s, :].T, dtype=np.float64)

        if p.shape != (n_actions, ids.size):
            raise ValueError(f"Probability vectors for state {s} have wrong shape.")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError(f"Invalid transition probabilities at state {s}.")
        mass = p.sum(axis=1)
        if not np.all((mass == 0) | np.isclose(mass, 1, rtol=1e-6, atol=1e-8)):
            raise ValueError(f"Each action at state {s} must have mass 0 or 1.")

        # No probability cutoff: even arbitrarily rare outcomes must be safe.
        positive = p > 0
        keep = positive.any(axis=0)
        rows.append((ids[keep], positive[:, keep]))

    width = max(1, max(len(ids) for ids, _ in rows))
    successors = np.zeros((n_states, width), dtype=np.intp)
    support = np.zeros((n_states, n_actions, width), dtype=bool)
    for s, (ids, positive) in enumerate(rows):
        successors[s, :len(ids)] = ids
        support[s, :, :len(ids)] = positive
    return successors, support
