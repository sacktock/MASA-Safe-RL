from __future__ import annotations
from typing import Any
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv


FAST_RATE = 0.9
SLOW_RATE = 0.1
OUT_RATE = 0.7
BUFFER_SIZE = 20
START_STATE = BUFFER_SIZE//2
EMPTY = 0

LABEL_DICT = defaultdict(set)
LABEL_DICT[START_STATE] = {"start"}
LABEL_DICT[EMPTY] = {"empty"} 

label_fn = lambda obs: LABEL_DICT[obs] 

cost_fn = lambda labels: 1.0 if "empty" in labels else 0.0

class MediaStreaming(TabularEnv):

    def __init__(self):
        super().__init__()

        self._fast_rate = FAST_RATE
        self._slow_rate = SLOW_RATE
        self._out_rate = OUT_RATE

        self._buffer_size = BUFFER_SIZE

        self._n_states = self._buffer_size
        self._n_actions = 2

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)

        self._transition_matrix = np.zeros((self._n_states, self._n_states, self._n_actions))
        for s in range(self._n_states):
            for a in range(self._n_actions):
                in_rate = self._slow_rate if a == 0 else self._fast_rate
                if s == 0:
                    self._transition_matrix[s+1, s, a] = in_rate * (1 - self._out_rate)
                    self._transition_matrix[s, s, a] = 1 - in_rate * (1 - self._out_rate)
                elif s == (self._n_states - 1):
                    self._transition_matrix[s-1, s, a] = (1 - in_rate) * self._out_rate
                    self._transition_matrix[s, s, a] = 1 - (1 - in_rate) * self._out_rate
                else:
                    self._transition_matrix[s+1, s, a] = in_rate * (1 - self._out_rate)
                    self._transition_matrix[s, s, a] = 1 - (in_rate * (1 - self._out_rate) + (1 - in_rate) * self._out_rate)
                    self._transition_matrix[s-1, s, a] = (1 - in_rate) * self._out_rate

        self._start_state = START_STATE

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

        reward = -1.0 if action == 1 else 0.0

        return self._state, reward, False, False, {}

    @property
    def safe_end_component(self):
        # This is a simplification - we assume the start states and above are safe, but every state has a small non-zero probability of being safe.
        return [s for s in range(self._start_state, self._n_states)]
        

