from __future__ import annotations
from typing import Any, Iterable, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from masa.common.label_fn import LabelFn
from collections import defaultdict 
from masa.envs.tabular.base import TabularEnv

START_STATE = 0
GOAL_STATE = 80
BLUE_STATE = 36
GREEN_STATE = 40
PURPLE_STATE = 4
SLIP_PROB = 0.1

label_dict = defaultdict(set)
label_dict[START_STATE] = {"start"}
label_dict[GOAL_STATE] = {"goal"}
label_dict[GREEN_STATE] = {"green"}
label_dict[PURPLE_STATE] = {"purple"}
label_dict[BLUE_STATE] = {"blue"}

label_fn = lambda obs: label_dict[obs] 

cost_fn = lambda labels: 1.0 if "blue" in labels else 0.0

def create_transition_matrix(grid_size: int, n_states: int, n_actions: int, slip_prob: float = 0.0):

    assert n_states == grid_size**2

    grid = np.zeros((grid_size, grid_size), dtype=int)
    for y in range(grid_size):
        grid[y] = np.arange(grid_size) + y*grid_size
    
    act_map = {0: (0, -1), # left
                1: (0, 1), # right
                2: (1, 0), # up
                3: (-1, 0), # down
                4: (0, 0), # stay
                5: (-1, -1), # left up
                6: (-1, 1), # left down
                7: (-1, 1), # right up
                8: (1, 1), # right down
                }

    assert n_actions < len(act_map.keys())
    matrix = np.zeros((n_states, n_states, n_actions))

    for y in range(grid_size):
        for x in range(grid_size):
            for a in range(n_actions):
                state = grid[y][x]
                next_y = int(np.clip(y + act_map[a][0], 0, grid_size-1))
                next_x = int(np.clip(x + act_map[a][1], 0, grid_size-1))
                next_state = grid[next_y, next_x]
                matrix[next_state, state, a] = 1.0 - slip_prob

                if slip_prob == 0.0:
                    continue

                rand_prob = slip_prob * 1 / (n_actions - 1)
                for rand_a in range(n_actions):
                    if rand_a == a:
                        continue
                    next_y = int(np.clip(y + act_map[rand_a][0], 0, grid_size-1))
                    next_x = int(np.clip(x + act_map[rand_a][1], 0, grid_size-1))
                    next_state = grid[next_y, next_x]
                    matrix[next_state, state, a] += rand_prob
    return matrix
            

class ColourGridWorld(TabularEnv):

    def __init__(self):
        super().__init__()

        self._grid_size = 9
        self._ncol = self._grid_size
        self._nrow = self._grid_size

        self._n_states = self._grid_size**2
        self._n_actions = 5

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)

        self._transition_matrix = create_transition_matrix(self._grid_size, self._n_states, self._n_actions, slip_prob=SLIP_PROB)

        self._start_state = START_STATE
        self._goal_state = GOAL_STATE

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

        terminated = bool(self._state == self._goal_state)
        reward = 1.0 if self._state == self._goal_state else 0.0

        return self._state, reward, terminated, False, {}


        
        

