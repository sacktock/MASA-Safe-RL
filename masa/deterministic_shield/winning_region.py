from __future__ import annotations

import numpy as np

def winning_region(
    targets: np.ndarray, support: np.ndarray, rejecting: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Greatest fixed point; all possible successors must remain winning.

    targets[q, s, k] is a flattened product-state ID.
    support[s, a, k] says whether that successor is possible under action a.
    rejecting[q] marks a bad-prefix DFA state. Padding has support=False.
    Returns winning[product_state] and allowed[product_state, action].
    """
    n_dfa, n_states, _ = targets.shape
    n_actions = support.shape[1]
    winning = np.broadcast_to(~rejecting[:, None], (n_dfa, n_states)).copy()
    enabled = support.any(axis=2)
    allowed = np.zeros((n_dfa, n_states, n_actions), dtype=bool)

    while True:
        flat = winning.ravel()
        for q in range(n_dfa):
            successor_wins = flat[targets[q]]
            escapes = (support & ~successor_wins[:, None, :]).any(axis=2)
            allowed[q] = winning[q, :, None] & enabled & ~escapes
        updated = allowed.any(axis=2)
        if np.array_equal(updated, winning):
            return winning.ravel(), allowed.reshape(-1, n_actions)
        winning = updated