from __future__ import annotations
from typing import Any, Literal
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.discrete.base import DiscreteEnv
from masa.envs.tabular.utils import create_pacman_transition_dict, create_pacman_end_component
from masa.envs.discrete.renderers.pacman import PacmanWithCoinsRenderer, validate_renderer_options
from masa.envs.tabular.renderers.pacman import PacmanHat, RGBColor

STANDARD_MAP = np.array([
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,0,1,0,1],
    [1,1,1,1,1,1,1,1,0,1]])
N_GHOSTS = 1
N_DIRECTIONS = 4
N_ACTIONS = 5
GHOST_RAND_PROB = 0.6
AGENT_START = (4, 1)
AGENT_TERM = (8, 6)
AGENT_DIRECTION = 1
GHOST_START = (3, 5)
GHOST_DIRECTION = 1

_, _, TRANSITION_MATRIX, N_STATES, STATE_MAP, REVERSE_STATE_MAP = \
create_pacman_transition_dict(
    STANDARD_MAP, 
    return_matrix=True, 
    n_directions=N_DIRECTIONS, 
    n_actions=N_ACTIONS, 
    n_ghosts=N_GHOSTS, 
    ghost_rand_prob=GHOST_RAND_PROB
)

def safety_abstraction(obs: np.ndarray) -> int:
    agent_slice = obs[:, :, 1 : 1 + N_DIRECTIONS]
    agent_y, agent_x, agent_direction = np.argwhere(agent_slice == 1)[0]
    start = 1 + N_DIRECTIONS
    end = start + N_DIRECTIONS
    ghost_slice = obs[:, :, start:end]
    ghost_y, ghost_x, ghost_direction = np.argwhere(ghost_slice == 1)[0]
    return STATE_MAP[(agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, 0)]

def abstr_label_fn(obs):
    (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, _) = REVERSE_STATE_MAP[obs]
    if (agent_y, agent_x) == (ghost_y, ghost_x):
        return {"ghost"}
    else:
        return set()

def label_fn(obs):
    agent_slice = obs[:, :, 1 : 1 + N_DIRECTIONS]
    agent_y, agent_x, _ = np.argwhere(agent_slice == 1)[0]
    start = 1 + N_DIRECTIONS
    end = start + N_DIRECTIONS
    ghost_slice = obs[:, :, start:end]
    ghost_y, ghost_x, _ = np.argwhere(ghost_slice == 1)[0]
    if (agent_y, agent_x) == (ghost_y, ghost_x):
        return {"ghost"}
    else:
        return set()

cost_fn = lambda labels: 1.0 if "ghost" in labels else 0.0
    
class MiniPacmanWithCoins(DiscreteEnv):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Literal["ansi", "rgb_array", "human"] | None = None,
        window_size: int = 512,
        pacman_hat: PacmanHat = "none",
        ghost_colors: tuple[RGBColor, ...] | None = None,
    ):
        super().__init__()
        validate_renderer_options(render_mode, window_size, pacman_hat)

        self._layout = STANDARD_MAP
        self._n_row = STANDARD_MAP.shape[0]
        self._n_col = STANDARD_MAP.shape[1]
        self._n_ghosts = N_GHOSTS
        self._n_directions = N_DIRECTIONS
        self._n_actions = N_ACTIONS

        self._transition_matrix = TRANSITION_MATRIX
        self._state_map = STATE_MAP
        self._reverse_state_map = REVERSE_STATE_MAP

        self._n_states = N_STATES
        self._n_actions = N_ACTIONS

        self._obs_shape = (self._n_row, self._n_col, self._n_directions*2 + 1)
        self.observation_space = spaces.Box(
            low=np.zeros(self._obs_shape, dtype=np.float32), 
            high=np.zeros(self._obs_shape, dtype=np.float32), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._n_actions)

        self._agent_start_x = AGENT_START[0]
        self._agent_start_y = AGENT_START[1]
        self._agent_start_direction = AGENT_DIRECTION

        self._ghost_start_x = GHOST_START[0]
        self._ghost_start_y = GHOST_START[1]
        self._ghost_start_direction = GHOST_DIRECTION
        self._agent_term_x = AGENT_TERM[0]
        self._agent_term_y = AGENT_TERM[1]

        self._start_state = self._state_map[
            (
                self._agent_start_y, 
                self._agent_start_x, 
                self._agent_start_direction, 
                self._ghost_start_y, 
                self._ghost_start_x, 
                self._ghost_start_direction,
                0
            )
        ]

        self.np_random = None
        self._state = None
        self._coin_array = None
        self._step_count = 0
        self.render_mode = render_mode
        self.window_size = int(window_size)
        self.pacman_hat = pacman_hat
        self.ghost_colors = ghost_colors
        self._renderer = PacmanWithCoinsRenderer(self)

    def _update_coin_array(self):
        (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, _) = self._reverse_state_map[self._state]
        self._coin_array[agent_y, agent_x] = 0.0

    def _obs(self):
        (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, _) = self._reverse_state_map[self._state]
        agent_loc = np.zeros((self._n_row, self._n_col, self._n_directions), dtype=np.float32)
        agent_loc[agent_y, agent_x, agent_direction] = 1.0
        ghost_loc = np.zeros((self._n_row, self._n_col, self._n_directions), dtype=np.float32)
        ghost_loc[ghost_y, ghost_x, ghost_direction] = 1.0
        return np.concatenate([self._coin_array[..., np.newaxis], agent_loc, ghost_loc], axis=-1, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed:
            self.np_random = np.random.default_rng(seed)

        if self.np_random is None:
            seed = np.random.SeedSequence().entropy
            self.np_random = np.random.default_rng(seed)

        self._coin_array = np.ones((self._n_row, self._n_col), dtype=np.float32)
        self._state = self._start_state
        self._step_count = 0
        obs = self._obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action: Any):
        assert self.action_space.contains(action), f"Invalid action {action}!"
        self._state = self.np_random.choice(self._n_states, p=self._transition_matrix[:, self._state, action])
        self._step_count += 1

        (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, _) = self._reverse_state_map[self._state]
        reward = float(self._coin_array[agent_y, agent_x])
        self._update_coin_array()

        terminated = True if (agent_x, agent_y) == AGENT_TERM else False

        obs = self._obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, {}

    def render(self):
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    @property
    def human_window_closed(self) -> bool:
        return self._renderer.human_window_closed

    def handle_pygame_event(self, event: Any) -> bool:
        return self._renderer.handle_pygame_event(event)

    @property
    def has_transition_matrix(self):
        return True
    
    @property
    def has_successor_states_dict(self):
        return False

    def get_transition_matrix(self):
        return self._transition_matrix

    def get_successor_states_dict(self):
        return None
