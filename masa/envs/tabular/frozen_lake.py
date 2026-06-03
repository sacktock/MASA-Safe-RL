from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import (
    FrozenLakeEnv as GymFrozenLakeEnv,
    generate_random_map,
)


DEFAULT_DESC = (
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
)

TILE_LABELS = {
    "S": {"start"},
    "F": {"frozen"},
    "H": {"hole"},
    "G": {"goal"},
}


def _normalize_desc(desc: Sequence[str] | np.ndarray | None) -> np.ndarray:
    if desc is None:
        desc = DEFAULT_DESC
    return np.asarray(desc, dtype="c")


def _tile_at(desc: Sequence[str] | np.ndarray, obs: int) -> str:
    flat_desc = _normalize_desc(desc).ravel()
    return bytes(flat_desc[int(obs)]).decode("utf-8")


def label_fn(obs: int) -> set[str]:
    """Default 4x4 FrozenLake labelling function."""
    return set(TILE_LABELS.get(_tile_at(DEFAULT_DESC, int(obs)), set()))


def cost_fn(labels: Iterable[str]) -> float:
    return 1.0 if "hole" in labels else 0.0


def _states_with_tile(desc: np.ndarray, tile: str) -> list[int]:
    return [
        int(i)
        for i, cell in enumerate(desc.ravel())
        if bytes(cell).decode("utf-8") == tile
    ]


def _build_transition_matrix(
    transition_table: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    matrix = np.zeros((n_states, n_states, n_actions), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            for probability, next_state, _, _ in transition_table[state][action]:
                matrix[int(next_state), state, action] += float(probability)
    return matrix


class FrozenLake(GymFrozenLakeEnv):
    """MASA-compatible FrozenLake environment with exact discrete dynamics.

    This class preserves Gymnasium's FrozenLake transition and rendering
    behavior while exposing a transition matrix and safety labels for MASA's
    discrete shield synthesis wrappers.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        desc: Sequence[str] | None = None,
        map_name: str | None = "4x4",
        is_slippery: bool = True,
        success_rate: float = 1.0 / 3.0,
        reward_schedule: tuple[float, float, float] = (1, 0, 0),
    ):
        super().__init__(
            render_mode=render_mode,
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery,
            success_rate=success_rate,
            reward_schedule=reward_schedule,
        )

        self._n_states = int(self.observation_space.n)
        self._n_actions = int(self.action_space.n)
        self._nrow, self._ncol = self.desc.shape
        self._grid_size = self._nrow

        self._start_state = int(np.flatnonzero(self.initial_state_distrib)[0])
        self._hole_states = _states_with_tile(self.desc, "H")
        self._goal_states = _states_with_tile(self.desc, "G")
        self._terminal_states = self._hole_states + self._goal_states

        self._transition_matrix = _build_transition_matrix(
            self.P,
            self._n_states,
            self._n_actions,
        )
        self._successor_states = None
        self._transition_probs = None

    def label_fn(self, obs: int) -> set[str]:
        return set(TILE_LABELS.get(_tile_at(self.desc, int(obs)), set()))

    def cost_fn(self, labels: Iterable[str]) -> float:
        return cost_fn(labels)

    @property
    def has_transition_matrix(self) -> bool:
        return self._transition_matrix is not None

    @property
    def has_successor_states_dict(self) -> bool:
        return False

    def get_transition_matrix(self) -> np.ndarray:
        return self._transition_matrix

    def get_successor_states_dict(self):
        return None

    def render(self) -> Any:
        if self.render_mode is None:
            return None
        return super().render()


__all__ = [
    "FrozenLake",
    "cost_fn",
    "generate_random_map",
    "label_fn",
]
