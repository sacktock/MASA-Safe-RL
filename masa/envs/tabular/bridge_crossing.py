from __future__ import annotations
from typing import Any
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv
from masa.envs.tabular.utils import create_transition_matrix

GRID_SIZE = 20
N_ACTIONS = 5
GRID = np.arange(GRID_SIZE**2).reshape(GRID_SIZE, GRID_SIZE)
START_STATE = int(GRID[-1, 0])
GOAL_STATES = list(GRID[:7, :].flatten())
LAVA_STATES = list(GRID[8:12, :8].flatten()) + list(GRID[8:12, -9:].flatten())
SLIP_PROB = 0.04

LABEL_DICT = defaultdict(set)
LABEL_DICT[START_STATE] = {"start"}
for _state in GOAL_STATES:
    LABEL_DICT[_state] = {"goal"}
for _state in LAVA_STATES:
    LABEL_DICT[_state] = {"lava"}

label_fn = lambda obs: LABEL_DICT[obs] 

cost_fn = lambda labels: 1.0 if "lava" in labels else 0.0

class BridgeCrossing(TabularEnv):


    def __init__(self):
        super().__init__()

        self._grid_size = GRID_SIZE
        self._ncol = self._grid_size
        self._nrow = self._grid_size

        self._n_states = self._grid_size**2
        self._n_actions = N_ACTIONS

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)

        self._start_state = START_STATE
        self._goal_states = GOAL_STATES
        self._lava_states = LAVA_STATES

        self._terminal_states = self._goal_states + self._lava_states

        self._transition_matrix = create_transition_matrix(self._grid_size, self._n_states, self._n_actions, slip_prob=SLIP_PROB, terminal_states=self._terminal_states)

        self.np_random = None
        self._state = None


    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed:
            self.np_random = np.random.default_rng(seed)

        if self.np_random is None:
            seed = np.random.SeedSequence().entropy
            self.np_random = np.random.default_rng(seed)

        self._state = self._start_state
        return self._state, {}


    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}!"
        self._state = self.np_random.choice(self._n_states, p=self._transition_matrix[:, self._state, action])

        terminated = bool(self._state in self._terminal_states)
        reward = 1.0 if self._state in self._goal_states else 0.0

        return self._state, reward, terminated, False, {}
