from __future__ import annotations
from typing import Any
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv
from masa.envs.tabular.utils import create_transition_matrix

GRID_SIZE = 15
N_ACTIONS = 5
START_STATES = [16,199,178,112,26]
GREEN_STATES = [170]
YELLOW_STATES = [176,191]
RED_STATES = [88,89,103,104]
BLUE_STATES = [121,122,136,137]
PINK_STATES = [53,54,68,69]
WALL_STATES = [45,60,75,210,195,180,165,150, 142] + \
    [211,212,213,214,215,216] + [220,221,222,223,224,209] + \
    [183,184,169,185,186,187] + [192,177,162,161,160] + \
    [143,144,129] + [138,139,140,141,125] + \
    [3,18,47,62,63,64,50,35,20,80,81,95] + \
    [83,84,99,100,116,131,133,134] + \
    [87,72,57,70,55,39,9,13,14,29,44,59]
BOMB_STATES = [76, 181, 123,82,207,8,58]
MEDIC_STATES = [154, 93, 38, 205, 74]
SLIP_PROB = 0.1

LABEL_DICT = defaultdict(set)
for _state in START_STATES:
    LABEL_DICT[_state] = {"start"}
for _state in GREEN_STATES:
    LABEL_DICT[_state] = {"green"}
for _state in YELLOW_STATES:
    LABEL_DICT[_state] = {"yellow"}
for _state in RED_STATES:
    LABEL_DICT[_state] = {"red"}
for _state in BLUE_STATES:
    LABEL_DICT[_state] = {"blue"}
for _state in PINK_STATES:
    LABEL_DICT[_state] = {"pink"}
for _state in BOMB_STATES:
    LABEL_DICT[_state] = {"bomb"}
for _state in MEDIC_STATES:
    LABEL_DICT[_state] = {"medic"}

label_fn = lambda obs: LABEL_DICT[obs] 

cost_fn = lambda labels: 1.0 if "bomb" in labels else 0.0

class ColourBombGridWorldV2(TabularEnv):

    def __init__(self):
        super().__init__()

        self._grid_size = GRID_SIZE
        self._ncol = self._grid_size
        self._nrow = self._grid_size

        self._n_states = self._grid_size**2
        self._n_actions = N_ACTIONS

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)

        self._start_states = START_STATES
        self._goal_states = GREEN_STATES + YELLOW_STATES + RED_STATES + BLUE_STATES + PINK_STATES
        self._safe_states = MEDIC_STATES

        self._transition_matrix = create_transition_matrix(self._grid_size, self._n_states, self._n_actions, slip_prob=SLIP_PROB, safe_states=self._safe_states, wall_states=WALL_STATES)

        # after reaching a goal state, transition to a random start state
        for _state in self._goal_states:
            for a in range(self._n_actions):
                probs = np.zeros_like(self._transition_matrix[:, _state, a])
                probs[self._start_states] = 1.0 / len(self._start_states)
                self._transition_matrix[:, _state, a] = probs

        self.np_random = None
        self._state = None


    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed:
            self.np_random = np.random.default_rng(seed)

        if self.np_random is None:
            seed = np.random.SeedSequence().entropy
            self.np_random = np.random.default_rng(seed)

        self._state = self.np_random.choice(self._start_states)
        return self._state, {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}!"
        self._state = self.np_random.choice(self._n_states, p=self._transition_matrix[:, self._state, action])

        reward = 1.0 if self._state in self._goal_states else 0.0

        return self._state, reward, False, False, {}