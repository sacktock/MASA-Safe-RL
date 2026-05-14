from __future__ import annotations
from typing import Any, Literal
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv
from masa.envs.tabular.utils import create_transition_matrix
from masa.envs.tabular.renderers.colour_bomb_grid_world import ColourBombGridWorldRenderer, validate_renderer_options

GRID_SIZE = 9
N_ACTIONS = 5
START_STATE = 74
GREEN_STATES = [65]
YELLOW_STATES = [70, 79]
BLUE_STATES = [9, 10, 18, 19]
PINK_STATES = [7, 8, 16, 17]
WALL_STATES = [11, 12, 13, 14, 15, 29, 30, 50, 52, 53, 55, 56, 57, 59, 64, 66, 69]
BOMB_STATES = [27, 43, 78]
SLIP_PROB = 0.1

LABEL_DICT = defaultdict(set)
LABEL_DICT[START_STATE] = {"start"}
for _state in GREEN_STATES:
    LABEL_DICT[_state] = {"green"}
for _state in YELLOW_STATES:
    LABEL_DICT[_state] = {"yellow"}
for _state in BLUE_STATES:
    LABEL_DICT[_state] = {"blue"}
for _state in PINK_STATES:
    LABEL_DICT[_state] = {"pink"}
for _state in BOMB_STATES:
    LABEL_DICT[_state] = {"bomb"}

label_fn = lambda obs: LABEL_DICT[obs] 

cost_fn = lambda labels: 1.0 if "bomb" in labels else 0.0

class ColourBombGridWorld(TabularEnv):
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
        self._start_states = [START_STATE]
        self._green_states = GREEN_STATES
        self._yellow_states = YELLOW_STATES
        self._red_states = []
        self._blue_states = BLUE_STATES
        self._pink_states = PINK_STATES
        self._wall_states = WALL_STATES
        self._bomb_states = BOMB_STATES
        self._medic_states = []
        self._goal_states = YELLOW_STATES + BLUE_STATES + PINK_STATES
        self._active_colour_dict = {}

        self._transition_matrix = create_transition_matrix(self._grid_size, self._n_states, self._n_actions, slip_prob=SLIP_PROB, terminal_states=self._goal_states, wall_states=WALL_STATES)

        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self._renderer = ColourBombGridWorldRenderer(self)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
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

        terminated = bool(self._state in self._goal_states)
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
