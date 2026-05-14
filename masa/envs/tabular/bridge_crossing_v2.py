from __future__ import annotations
from typing import Any, Literal
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv
from masa.envs.tabular.utils import create_transition_matrix
from masa.envs.tabular.renderers.bridge_crossing import BridgeCrossingRenderer, validate_renderer_options

GRID_SIZE = 20
N_ACTIONS = 5
GRID = np.arange(GRID_SIZE**2).reshape(GRID_SIZE, GRID_SIZE)
START_STATE = int(GRID[-1, 0])
GOAL_STATES = list(GRID[:7, :].flatten())
LAVA_STATES = list(GRID[8:12, 2:16].flatten()) + list([GRID[11, 1]])
SLIP_PROB = 0.04

LABEL_DICT = defaultdict(set)
LABEL_DICT[START_STATE] = {"start"}
for _state in GOAL_STATES:
    LABEL_DICT[_state] = {"goal"}
for _state in LAVA_STATES:
    LABEL_DICT[_state] = {"lava"}

label_fn = lambda obs: LABEL_DICT[obs] 

cost_fn = lambda labels: 1.0 if "lava" in labels else 0.0

class BridgeCrossingV2(TabularEnv):
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
        self._goal_states = GOAL_STATES
        self._lava_states = LAVA_STATES

        self._terminal_states = self._goal_states + self._lava_states

        self._transition_matrix = create_transition_matrix(self._grid_size, self._n_states, self._n_actions, slip_prob=SLIP_PROB, terminal_states=self._terminal_states)

        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self._renderer = BridgeCrossingRenderer(self)


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

        terminated = bool(self._state in self._terminal_states)
        reward = 1.0 if self._state in self._goal_states else 0.0

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
