from __future__ import annotations
from typing import Any
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.discrete.base import DiscreteEnv
from masa.envs.tabular.utils import create_pacman_transition_dict

STANDARD_MAP = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
N_GHOSTS = 1
N_DIRECTIONS = 4
N_ACTIONS = 5
GHOST_RAND_PROB = 0.6
AGENT_START = (1, 7)
AGENT_DIRECTION = 1
GHOST_START = (12, 7)
GHOST_DIRECTION = 3

SUCCESSOR_STATES, TRANSITION_PROBS, _, N_STATES, STATE_MAP, REVERSE_STATE_MAP = \
create_pacman_transition_dict(
    STANDARD_MAP, 
    return_matrix=False, 
    n_directions=N_DIRECTIONS, 
    n_actions=N_ACTIONS, 
    n_ghosts=N_GHOSTS, 
    ghost_rand_prob=GHOST_RAND_PROB
)

def label_fn(obs):
    (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, _) = REVERSE_STATE_MAP[obs]
    if (agent_y, agent_x) == (ghost_y, ghost_x):
        return {"ghost"}
    else:
        return set()

cost_fn = lambda labels: 1.0 if "ghost" in labels else 0.0
    
class Pacman(TabularEnv):

    def __init__(self):


        self._n_row = STANDARD_MAP.shape[0]
        self._n_col = STANDARD_MAP.shape[1]
        self._n_ghosts = N_GHOSTS
        self._n_directions = N_DIRECTIONS
        self._n_actions = N_ACTIONS

        self._successor_states = SUCCESSOR_STATES
        self._transition_probs = TRANSITION_PROBS
        self._state_map = STATE_MAP
        self._reverse_state_map = REVERSE_STATE_MAP

        self._n_states = N_STATES
        self._n_actions = N_ACTIONS

        self._obs_shape = (self._n_row, self._n_col, self._n_directions*2 + 1)
        self.observation_space = spaces.Box(low=np.zeros(self._obs_shape), high=np.zeros(self._obs_shape), dtype=np.float32)
        self.action_space = spaces.Discrete(self._n_actions)

        self._agent_start_x = AGENT_START[0]
        self._agent_start_y = AGENT_START[1]
        self._agent_start_direction = AGENT_DIRECTION

        self._ghost_start_x = GHOST_START[0]
        self._ghost_start_y = GHOST_START[1]
        self._ghost_start_direction = GHOST_DIRECTION

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
        return self._obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}!"
        self._state = self.np_random.choice(self._successor_states[self._state], p=self._transition_probs[(self._state, action)])

        (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, _) = self._reverse_state_map[self._state]
        reward = float(self._coin_array[agent_y, agent_x])
        self._update_coin_array()

        return self._obs(), reward, False, False, {}

    @property
    def safe_end_component(self):
        return []
