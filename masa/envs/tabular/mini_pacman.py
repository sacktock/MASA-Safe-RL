from __future__ import annotations
from typing import Any
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv
from masa.envs.tabular.utils import create_pacman_transition_dict, create_pacman_end_component

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
FOOD = (7, 3)
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
    ghost_rand_prob=GHOST_RAND_PROB, 
    food_x=FOOD[0], 
    food_y=FOOD[1]
)

def label_fn(obs):
    (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food) = REVERSE_STATE_MAP[obs]
    if (agent_y == FOOD[1]) and (agent_x == FOOD[0]) and (not ((agent_y, agent_x) == (ghost_y, ghost_x))) and food:
        return {"food"}
    elif (agent_y, agent_x) == (ghost_y, ghost_x):
        return {"ghost"}
    else:
        return set()

cost_fn = lambda labels: 1.0 if "ghost" in labels else 0.0
    
class MiniPacman(TabularEnv):

    def __init__(self):
        super().__init__()

        self._n_row = STANDARD_MAP.shape[0]
        self._n_col = STANDARD_MAP.shape[1]
        self._n_ghosts = N_GHOSTS
        self._n_directions = N_DIRECTIONS
        self._n_actions = N_ACTIONS

        self._food_x = FOOD[0]
        self._food_y = FOOD[1]

        self._transition_matrix = TRANSITION_MATRIX
        self._state_map = STATE_MAP
        self._reverse_state_map = REVERSE_STATE_MAP

        self._n_states = N_STATES
        self._n_actions = N_ACTIONS

        self.observation_space = spaces.Discrete(self._n_states)
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
                1
            )
        ]

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

        (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food) = self._reverse_state_map[self._state]
        if (agent_y == self._food_y) and (agent_x == self._food_x) and (not ((agent_y, agent_x) == (ghost_y, ghost_x))) and food:
            reward = 1.0
        else:
            reward = 0.0

        terminated = True if (agent_x, agent_y) == AGENT_TERM else False

        return self._state, reward, terminated, False, {}
