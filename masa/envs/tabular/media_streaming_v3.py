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
DANGER_THRESHOLD = 20
DANGER_CAP = DANGER_THRESHOLD + 1  # represents all danger levels > DANGER_THRESHOLD
DANGER_STATES = DANGER_CAP + 1
TIME_STATES = EPISODE_LENGTH + 1
SAFETY_STATES = DANGER_STATES * TIME_STATES
SLOW_DANGER_DOWN_PROB = 0.5
SLOW_DANGER_UP_PROB = 0.4
FAST_DANGER_UP_PROB = 0.8
FAST_DANGER_DOWN_PROB = 0.1
START_BUFFER = BUFFER_SIZE // 2
START_DANGER = 0
START_TIME = 0
EMPTY = 0


def _encode_safety_state(danger_level: int, time_step: int = START_TIME) -> int:
    return time_step * DANGER_STATES + danger_level


def _decode_safety_state(state: int) -> tuple[int, int]:
    danger_level = state % DANGER_STATES
    time_step = state // DANGER_STATES
    return danger_level, time_step


def _state_to_obs(buffer_level: int, danger_level: int, time_step: int) -> dict[str, int]:
    return {
        "danger": danger_level,
        "buffer": buffer_level,
        "time": time_step,
    }


def safety_abstraction(obs: Any) -> int:
    if np.isscalar(obs):
        return int(obs)

    if isinstance(obs, dict):
        danger_level = int(obs["danger"])
        time_step = int(obs["time"])
    else:
        obs = np.asarray(obs)
        if obs.ndim == 0:
            return int(obs)

        danger_level = int(round(float(obs[0])))
        time_step = int(round(float(obs[2] if obs.shape[0] > 2 else obs[1])))

    danger_level = int(np.clip(danger_level, 0, DANGER_CAP))
    time_step = int(np.clip(time_step, 0, EPISODE_LENGTH))
    return _encode_safety_state(danger_level, time_step)


def label_fn(obs: Any) -> set[str]:
    buffer_level = int(obs["buffer"]) if isinstance(obs, dict) else None
    danger_level, time_step = _decode_safety_state(safety_abstraction(obs))
    labels = set()
    if danger_level == START_DANGER and time_step == START_TIME:
        labels.add("start")
    if buffer_level == EMPTY:
        labels.add("empty")
    if danger_level >= DANGER_CAP:
        labels.add("unsafe")
    return labels


cost_fn = lambda labels: 1.0 if "unsafe" in labels else 0.0


class MediaStreamingV3(TabularEnv):

    def __init__(self):
        super().__init__()

        self._fast_rate = FAST_RATE
        self._slow_rate = SLOW_RATE
        self._out_rate = OUT_RATE
        self._episode_length = EPISODE_LENGTH
        self._danger_threshold = DANGER_THRESHOLD
        self._danger_cap = DANGER_CAP
        self._slow_danger_down_prob = SLOW_DANGER_DOWN_PROB
        self._slow_danger_up_prob = SLOW_DANGER_UP_PROB
        self._fast_danger_up_prob = FAST_DANGER_UP_PROB
        self._fast_danger_down_prob = FAST_DANGER_DOWN_PROB

        self._buffer_size = BUFFER_SIZE

        self._n_states = SAFETY_STATES
        self._n_actions = 2

        self.observation_space = spaces.Dict({
            "danger": spaces.Discrete(DANGER_STATES),
            "buffer": spaces.Discrete(self._buffer_size),
            "time": spaces.Discrete(TIME_STATES),
        })
        self.action_space = spaces.Discrete(self._n_actions)

        self._successor_states = {}
        self._transition_probs = {}
        for s in range(self._n_states):
            danger_level, time_step = _decode_safety_state(s)
            probs_by_action = []
            for a in range(self._n_actions):
                if time_step >= self._episode_length:
                    probs_by_action.append({s: 1.0})
                    continue

                danger_transitions = self._danger_transitions(danger_level, a)
                next_time_step = time_step + 1

                probs_by_successor = {}
                for next_danger_level, danger_prob in danger_transitions:
                    next_state = _encode_safety_state(next_danger_level, next_time_step)
                    probs_by_successor[next_state] = (
                        probs_by_successor.get(next_state, 0.0) + danger_prob
                    )
                probs_by_action.append(probs_by_successor)

            successors = sorted(set().union(*[set(probs.keys()) for probs in probs_by_action]))
            self._successor_states[s] = successors
            for a, probs_by_successor in enumerate(probs_by_action):
                self._transition_probs[(s, a)] = [
                    probs_by_successor.get(successor, 0.0) for successor in successors
                ]

        self._start_state = _encode_safety_state(START_DANGER, START_TIME)

        self.np_random = None
        self._buffer_level = None
        self._danger_level = None
        self._time_step = None

    def _buffer_transitions(self, buffer_level: int, action: int) -> list[tuple[int, float]]:
        in_rate = self._slow_rate if action == 0 else self._fast_rate

        p_up = in_rate * (1 - self._out_rate)
        p_down = (1 - in_rate) * self._out_rate

        if buffer_level == 0:
            return [
                (buffer_level + 1, p_up),
                (buffer_level, 1.0 - p_up),
            ]

        if buffer_level == (self._buffer_size - 1):
            return [
                (buffer_level - 1, p_down),
                (buffer_level, 1.0 - p_down),
            ]

        return [
            (buffer_level + 1, p_up),
            (buffer_level - 1, p_down),
            (buffer_level, 1.0 - (p_up + p_down)),
        ]

    def _danger_transitions(self, danger_level: int, action: int) -> list[tuple[int, float]]:
        if action == 0:
            return self._combine_duplicate_transitions([
                (max(0, danger_level - 1), self._slow_danger_down_prob),
                (danger_level, 1.0 - self._slow_danger_down_prob - self._slow_danger_up_prob),
                (min(self._danger_cap, danger_level + 1), self._slow_danger_up_prob),
            ])

        return self._combine_duplicate_transitions([
            (min(self._danger_cap, danger_level + 1), self._fast_danger_up_prob),
            (danger_level, 1.0 - self._fast_danger_up_prob - self._fast_danger_down_prob),
            (max(0, danger_level - 1), self._fast_danger_down_prob),
        ])

    def _combine_duplicate_transitions(self, transitions: list[tuple[int, float]]) -> list[tuple[int, float]]:
        combined = {}
        for state, probability in transitions:
            combined[state] = combined.get(state, 0.0) + probability
        return [(state, probability) for state, probability in combined.items() if probability > 0.0]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed:
            self.np_random = np.random.default_rng(seed)

        if self.np_random is None:
            seed = np.random.SeedSequence().entropy
            self.np_random = np.random.default_rng(seed)

        self._buffer_level = START_BUFFER
        self._danger_level = START_DANGER
        self._time_step = START_TIME
        return _state_to_obs(self._buffer_level, self._danger_level, self._time_step), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}!"
        if self._time_step < self._episode_length:
            buffer_transitions = self._buffer_transitions(self._buffer_level, action)
            next_buffers, buffer_probs = zip(*buffer_transitions)
            self._buffer_level = int(self.np_random.choice(next_buffers, p=buffer_probs))

            danger_transitions = self._danger_transitions(self._danger_level, action)
            next_dangers, danger_probs = zip(*danger_transitions)
            self._danger_level = int(self.np_random.choice(next_dangers, p=danger_probs))
            self._time_step += 1

        reward = -1.0 if self._buffer_level == EMPTY else 0.0

        return _state_to_obs(self._buffer_level, self._danger_level, self._time_step), reward, False, False, {}
