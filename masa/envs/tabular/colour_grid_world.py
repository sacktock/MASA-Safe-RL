from __future__ import annotations
from typing import Any, Literal
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv
from masa.envs.tabular.utils import create_transition_matrix
from masa.envs.tabular.renderers.colour_grid_world import ColourGridWorldRenderer, validate_renderer_options

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
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Literal["ansi", "rgb_array", "human"] | None = None,
        render_window_size: int = 512,
    ):
        super().__init__()
        validate_renderer_options(render_mode, render_window_size)

        self._grid_size = GRID_SIZE
        self._ncol = self._grid_size
        self._nrow = self._grid_size

        self._n_states = self._grid_size**2
        self._n_actions = N_ACTIONS

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)

        self._start_state = START_STATE
        self._goal_state = GOAL_STATE
        self._blue_state = BLUE_STATE
        self._green_state = GREEN_STATE
        self._purple_state = PURPLE_STATE

        self._transition_matrix = create_transition_matrix(self._grid_size, self._n_states, self._n_actions, slip_prob=SLIP_PROB, terminal_states=[self._goal_state])

        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self._renderer = ColourGridWorldRenderer(self)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed:
            self.np_random = np.random.default_rng(seed)

        if self.np_random is None:
            seed = np.random.SeedSequence().entropy
            self.np_random = np.random.default_rng(seed)

        self._state = self._start_state
        self._step_count = 0
        if self.render_mode == "human":
            self.render()
        return self._state, {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}!"
        self._state = self.np_random.choice(self._n_states, p=self._transition_matrix[:, self._state, action])
        self._step_count += 1

        terminated = bool(self._state == self._goal_state)
        reward = 1.0 if self._state == self._goal_state else 0.0

        if self.render_mode == "human":
            self.render()
        return self._state, reward, terminated, False, {}

    def render(self):
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    @property
    def human_window_closed(self) -> bool:
        return self._renderer.human_window_closed

    def handle_pygame_event(self, event: Any) -> bool:
        return self._renderer.handle_pygame_event(event)
