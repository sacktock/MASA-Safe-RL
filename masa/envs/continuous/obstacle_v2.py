from __future__ import annotations
from typing import Any, Literal
from gymnasium import spaces
import numpy as np
import math
from masa.common.label_fn import LabelFn
from masa.envs.continuous.base import ContinuousEnv
from masa.envs.continuous.renderers.obstacle import ObstacleRenderer, validate_renderer_options

OBSTACLES = [
    np.array([
        [1.0, 2.0], # in the interval [1, 2] on the x axis 
        [1.0, 2.0], # in the interval [1, 2] on the y axis
    ]),
]

GOAL_POSITION = np.array([3.0, 3.0])
MAX_SPEED =  0.05
MIN_POSITION = -0.5
MAX_POSITION = 3.5

def label_fn(obs):
    labels = set()
    position, velocity = obs[:2], obs[2:]
    for obstacle in OBSTACLES:
        lower = obstacle[:, 0]
        upper = obstacle[:, 1]
        if np.all(position >= lower) and np.all(position <= upper):
            labels.add("obstacle")

    if np.all(position >= GOAL_POSITION):
        labels.add("goal")

    if np.any(position >= MAX_POSITION) or np.any(position <= MIN_POSITION):
        labels.add("boundary")

    if np.any(np.abs(velocity) >= MAX_SPEED):
        labels.add("max_speed")

    return labels

cost_fn = lambda labels: 1.0 if "obstacle" in labels else 0.0

class ObstacleV2(ContinuousEnv):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Literal["ansi", "rgb_array", "human"] | None = None,
        render_window_size: int = 512,
    ):
        validate_renderer_options(render_mode, render_window_size)

        self._dt = 1.0
        self._power = 0.001
        self._max_speed = MAX_SPEED
        self._min_position = MIN_POSITION
        self._max_position = MAX_POSITION

        self._goal_position = GOAL_POSITION 

        self._obstacles = OBSTACLES

        low = np.array(
            [
                self._min_position,
                self._min_position,
                -self._max_speed,
                -self._max_speed,
            ],
            dtype=np.float32
        )

        high = np.array(
            [
                self._max_position,
                self._max_position,
                self._max_speed,
                self._max_speed,
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-2.0, -2.0]), high=np.array([2.0, 2.0]), shape=(2,), dtype=np.float32)

        self.np_random = None
        self._state = None
        self._step_count = 0
        self._last_action = None

        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self._renderer = ObstacleRenderer(self)

    def _obs(self):
        return self._state

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed:
            self.np_random = np.random.default_rng(seed)

        if self.np_random is None:
            seed = np.random.SeedSequence().entropy
            self.np_random = np.random.default_rng(seed)

        self._state = np.array(
            [
                self.np_random.uniform(low=-0.05, high=-0.05),
                self.np_random.uniform(low=-0.05, high=-0.05),
                0.0,
                0.0
            ],
            dtype=np.float32
        )

        self._step_count = 0
        self._last_action = None
        if self.render_mode == "human":
            self.render()

        return self._obs(), {}

    def step(self, action: Any):
        assert self.action_space.contains(action), f"Invalid action {action}!"
        state = self._state
        velocity = state[2:] + 5.0 * action * self._power
        velocity = np.clip(velocity, -self._max_speed, self._max_speed)
        position = state[:2] + 2.0 * velocity * self._dt
        position = np.clip(position, self._min_position, self._max_position)
        self._state = np.concatenate([position, velocity])
        self._step_count += 1
        self._last_action = np.array(action, dtype=np.float32)

        reward = 30.0 if np.all(position >= self._goal_position) else \
            (np.linalg.norm(state[:2] - self._goal_position) - np.linalg.norm(position - self._goal_position))
        terminal = np.all(position >= self._goal_position)

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminal, False, {}

    def render(self):
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    @property
    def human_window_closed(self) -> bool:
        return self._renderer.human_window_closed

    def handle_pygame_event(self, event: Any) -> bool:
        return self._renderer.handle_pygame_event(event)
