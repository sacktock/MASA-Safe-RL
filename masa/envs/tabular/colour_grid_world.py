from __future__ import annotations
from typing import Any
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv
from masa.envs.tabular.utils import create_transition_matrix

GRID_SIZE = 9
N_ACTIONS = 5
START_STATE = 0
GOAL_STATE = 80
BLUE_STATE = 36
GREEN_STATE = 40
PURPLE_STATE = 4
SLIP_PROB = 0.1

LABEL_DICT = defaultdict(set)
LABEL_DICT[START_STATE] = {"start"}
LABEL_DICT[GOAL_STATE] = {"goal"}
LABEL_DICT[GREEN_STATE] = {"green"}
LABEL_DICT[PURPLE_STATE] = {"purple"}
LABEL_DICT[BLUE_STATE] = {"blue"}

label_fn = lambda obs: LABEL_DICT[obs] 

cost_fn = lambda labels: 1.0 if "blue" in labels else 0.0

class ColourGridWorld(TabularEnv):

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
        self._goal_state = GOAL_STATE

        self._transition_matrix = create_transition_matrix(self._grid_size, self._n_states, self._n_actions, slip_prob=SLIP_PROB, terminal_states=[self._goal_state])

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

        terminated = bool(self._state == self._goal_state)
        reward = 1.0 if self._state == self._goal_state else 0.0

        return self._state, reward, terminated, False, {}


        
        

