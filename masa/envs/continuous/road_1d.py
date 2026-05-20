from __future__ import annotations
from typing import Any, Literal
from gymnasium import spaces
import numpy as np
import math
from masa.common.label_fn import LabelFn
from masa.envs.continuous.base import ContinuousEnv
from masa.envs.continuous.renderers.road_1d import Road1DRenderer, validate_renderer_options

GOAL_POSITION = 3.0
MAX_SPEED = 0.1
SPEED_LIMIT = 0.01
MIN_POSITION = -4.0
MAX_POSITION = 4.0

def label_fn(obs):
    labels = set()
    position, velocity = obs
    if position >= GOAL_POSITION:
        labels.add("goal")
    if position >= MAX_POSITION or position <= MIN_POSITION:
        labels.add("boundary")
    if np.abs(velocity) >= MAX_SPEED:
        labels.add("max_speed")
    if np.abs(velocity) >= SPEED_LIMIT:
        labels.add("over")
    return labels
    
cost_fn = lambda labels: 1.0 if "over" in labels else 0.0

class Road1D(ContinuousEnv):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 25}

    def __init__(
        self,
        render_mode: Literal["ansi", "rgb_array", "human"] | None = None,
        render_window_size: int = 512,
    ):
        validate_renderer_options(render_mode, render_window_size)

        self._dt = 1.0
        self._power = 0.0001
        self._max_speed = MAX_SPEED
        self._min_position = MIN_POSITION
        self._max_position = MAX_POSITION

        self._goal_position = GOAL_POSITION

        self._speed_limit = SPEED_LIMIT

        low = np.array(
            [
                self._min_position,
                -self._max_speed,
            ],
            dtype=np.float32
        )

        high = np.array(
            [
                self._max_position,
                self._max_speed,
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

        self.np_random = None
        self._state = None
        self._step_count = 0
        self._last_action = None

        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self._renderer = Road1DRenderer(self)

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
                0.0,
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
        velocity = state[1] + 10.0 * action[0] * self._power
        velocity = np.clip(velocity, -self._max_speed, self._max_speed)
        position = state[0] + 10.0 * velocity * self._dt
        position = np.clip(position, self._min_position, self._max_position)
        self._state = np.array([position, velocity], dtype=np.float32)
        self._step_count += 1
        self._last_action = np.array(action, dtype=np.float32)

        reward = 20.0 if position >= self._goal_position else position - state[0] 
            
        terminal = position >= self._goal_position

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
