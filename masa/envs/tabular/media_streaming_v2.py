from __future__ import annotations
from typing import Any, Dict
from gymnasium import spaces
import numpy as np
from masa.envs.tabular.base import TabularEnv


FAST_RATE = 0.9
SLOW_RATE = 0.1
OUT_RATE = 0.7
BUFFER_SIZE = 20
EPISODE_LENGTH = 100
C_THRESHOLD = EPISODE_LENGTH // 2
FAST_COUNT_CAP = C_THRESHOLD + 1  # represents all counts > C_THRESHOLD
FAST_COUNT_STATES = FAST_COUNT_CAP + 1
START_BUFFER = BUFFER_SIZE // 2
EMPTY = 0

def _encode_state(buffer_level: int, fast_count: int) -> int:
    return fast_count * BUFFER_SIZE + buffer_level

def _decode_state(state: int) -> tuple[int, int]:
    fast_count = state // BUFFER_SIZE
    buffer_level = state % BUFFER_SIZE
    return buffer_level, fast_count

def label_fn(obs: int) -> set[str]:
    buffer_level, fast_count = _decode_state(obs)
    labels = set()
    if buffer_level == START_BUFFER and fast_count == 0:
        labels.add("start")
    if buffer_level == EMPTY:
        labels.add("empty")
    if fast_count >= FAST_COUNT_CAP:
        labels.add("unsafe")
    return labels

cost_fn = lambda labels: 1.0 if "unsafe" in labels else 0.0

class MediaStreamingV2(TabularEnv):

    def __init__(self):
        super().__init__()

        self._fast_rate = FAST_RATE
        self._slow_rate = SLOW_RATE
        self._out_rate = OUT_RATE
        self._episode_length = EPISODE_LENGTH
        self._c_threshold = C_THRESHOLD
        self._fast_count_cap = FAST_COUNT_CAP

        self._buffer_size = BUFFER_SIZE

        self._n_states = self._buffer_size * (self._fast_count_cap + 1)
        self._n_actions = 2

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)

        self._transition_matrix = np.zeros((self._n_states, self._n_states, self._n_actions))
        for s in range(self._n_states):
            buffer_level, fast_count = _decode_state(s)
            for a in range(self._n_actions):
                in_rate = self._slow_rate if a == 0 else self._fast_rate
                next_fast_count = min(
                    self._fast_count_cap,
                    fast_count + (1 if a == 1 else 0),
                )

                p_up = in_rate * (1 - self._out_rate)
                p_down = (1 - in_rate) * self._out_rate

                if buffer_level == 0:
                    self._transition_matrix[_encode_state(buffer_level + 1, next_fast_count), s, a] += p_up
                    self._transition_matrix[_encode_state(buffer_level, next_fast_count), s, a] += 1.0 - p_up
                elif buffer_level == (self._buffer_size - 1):
                    self._transition_matrix[_encode_state(buffer_level - 1, next_fast_count), s, a] += p_down
                    self._transition_matrix[_encode_state(buffer_level, next_fast_count), s, a] += 1.0 - p_down
                else:
                    self._transition_matrix[_encode_state(buffer_level + 1, next_fast_count), s, a] += p_up
                    self._transition_matrix[_encode_state(buffer_level - 1, next_fast_count), s, a] += p_down
                    self._transition_matrix[_encode_state(buffer_level, next_fast_count), s, a] += 1.0 - (p_up + p_down)

        self._start_state = _encode_state(START_BUFFER, 0)

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

        buffer_level, _ = _decode_state(self._state)
        reward = -1.0 if buffer_level == EMPTY else 0.0

        return self._state, reward, False, False, {}
        

