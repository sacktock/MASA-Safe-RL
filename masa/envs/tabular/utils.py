from __future__ import annotations
from typing import List, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def create_transition_matrix(
    grid_size: int, 
    n_states: int, 
    n_actions: int, 
    slip_prob: float = 0.0, 
    terminal_states: List[int] = [], 
    safe_states: List[int] = [], 
    wall_states: List[int] = []
    ):

    assert n_states == grid_size**2

    grid = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    
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
                if state in terminal_states:
                    matrix[state, state, a] = 1.0
                    continue

                next_y = int(np.clip(y + act_map[a][0], 0, grid_size-1))
                next_x = int(np.clip(x + act_map[a][1], 0, grid_size-1))
                next_state = grid[next_y, next_x]

                next_state = state if next_state in wall_states else next_state

                p = 1.0 if state in safe_states else 1.0 - slip_prob
                matrix[next_state, state, a] += p

                if p == 1.0:
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

def create_advanced_transition_matrix(
    grid_size: int, 
    n_coloured_zones: int, 
    n_states: int, 
    n_actions: int, 
    coloured_states: int, 
    slip_prob: float = 0.0, 
    terminal_states: List[int] = [], 
    safe_states: List[int] = [], 
    wall_states: List[int] = []
    ):

    assert n_states == (grid_size**2)*n_coloured_zones

    grid = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)

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

    for i in range(n_coloured_zones):
        z = i * (grid_size**2)
        for y in range(grid_size):
            for x in range(grid_size):
                state = grid[y][x] + z
                for a in range(n_actions):
                    if state in terminal_states:
                        matrix[state, state, a] = 1.0
                        continue

                    next_y = int(np.clip(y + act_map[a][0], 0, grid_size-1))
                    next_x = int(np.clip(x + act_map[a][1], 0, grid_size-1))
                    next_state = grid[next_y, next_x] + z

                    next_state = state if next_state in wall_states else next_state

                    p = 1.0 if state in safe_states else 1.0 - slip_prob

                    if state in coloured_states:
                        probs = (p / (n_coloured_zones -1))
                        for c in range(n_coloured_zones):
                            if c == i:
                                continue
                            matrix[next_state - z + c * (grid_size**2), state, a] += probs
                    else:
                        matrix[next_state, state, a] += p

                    if p == 1.0:
                        continue

                    rand_prob = slip_prob * 1 / (n_actions - 1)
                    for rand_a in range(n_actions):
                        if rand_a == a:
                            continue

                        next_y = int(np.clip(y + act_map[rand_a][0], 0, grid_size-1))
                        next_x = int(np.clip(x + act_map[rand_a][1], 0, grid_size-1))
                        next_state = grid[next_y, next_x] + z

                        next_state = state if next_state in wall_states else next_state

                        if state in coloured_states:
                            probs = (rand_prob / (n_coloured_zones -1))
                            for c in range(n_coloured_zones):
                                if c == i:
                                    continue
                                matrix[next_state - z + c * (grid_size**2), state, a] += probs
                        else:
                            matrix[next_state, state, a] += rand_prob

    return matrix

def create_pacman_transition_dict(
    standard_map: np.array, 
    return_matrix: bool = False, 
    n_directions: int = 4, 
    n_actions: int = 5, 
    n_ghosts: int = 1, 
    ghost_rand_prob: float = 0.6, 
    food_x: Optional[int] = None, 
    food_y: Optional[int] = None
):

    nrow = standard_map.shape[0]
    ncol = standard_map.shape[1]

    grid = np.arange(nrow*ncol).reshape(nrow, ncol)

    assert n_ghosts == 1, f"function only supports n_ghosts=1 not {n_ghosts}"

    action_map = {0: (0, -1), # left
                  1: (0, 1), # right
                  2: (1, 0), # down
                  3: (-1, 0), # up
                  4: (0, 0), # stay
                  }

    direction_map = {0: (0, -1), # left
                     1: (0, 1), # right
                     2: (1, 0), # down
                     3: (-1, 0), # up
                    }

    reverse_map = {0: 1,
                   1: 0,
                   2: 3,
                   3: 2,}

    assert n_actions <= len(action_map)
    assert n_directions <= len(direction_map)

    state_map = {}
    reverse_state_map = {}

    state_idx = 0

    # enumerate the valid states
    for agent_y in range(nrow):
        for agent_x in range(ncol):
            for agent_direction in range(n_directions):
                
                infront_agent_y = int(np.clip(agent_y + direction_map[agent_direction][0], 0, nrow-1))
                infront_agent_x = int(np.clip(agent_x + direction_map[agent_direction][1], 0, ncol-1))

                behind_agent_y = int(np.clip(agent_y - direction_map[agent_direction][0], 0, nrow-1))
                behind_agent_x = int(np.clip(agent_x - direction_map[agent_direction][1], 0, ncol-1))

                if standard_map[infront_agent_y, infront_agent_x] == 1 and standard_map[behind_agent_y, behind_agent_x] == 1:
                    continue

                for ghost_y in range(nrow):
                    for ghost_x in range(ncol):
                        for ghost_direction in range(n_directions):
                            if (food_x is not None) and (food_y is not None):
                                for food in [0, 1]:
                                    ghost_loc = grid[ghost_y, ghost_x]
                                    infront_ghost_y = int(np.clip(ghost_y + direction_map[ghost_direction][0], 0, nrow-1))
                                    infront_ghost_x = int(np.clip(ghost_x + direction_map[ghost_direction][1], 0, ncol-1))

                                    behind_ghost_y = int(np.clip(ghost_y - direction_map[ghost_direction][0], 0, nrow-1))
                                    behind_ghost_x = int(np.clip(ghost_x - direction_map[ghost_direction][1], 0, ncol-1))

                                    if standard_map[infront_ghost_y, infront_ghost_x] == 1 and standard_map[behind_ghost_y, behind_ghost_x] == 1:
                                        continue

                                    if standard_map[agent_y, agent_x] == 0 and standard_map[ghost_y, ghost_x] == 0:
                                        state_map[(agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food)] = state_idx
                                        reverse_state_map[state_idx] = (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food)
                                        state_idx += 1
                            else:
                                ghost_loc = grid[ghost_y, ghost_x]
                                infront_ghost_y = int(np.clip(ghost_y + direction_map[ghost_direction][0], 0, nrow-1))
                                infront_ghost_x = int(np.clip(ghost_x + direction_map[ghost_direction][1], 0, ncol-1))

                                behind_ghost_y = int(np.clip(ghost_y - direction_map[ghost_direction][0], 0, nrow-1))
                                behind_ghost_x = int(np.clip(ghost_x - direction_map[ghost_direction][1], 0, ncol-1))

                                if standard_map[infront_ghost_y, infront_ghost_x] == 1 and standard_map[behind_ghost_y, behind_ghost_x] == 1:
                                    continue

                                if standard_map[agent_y, agent_x] == 0 and standard_map[ghost_y, ghost_x] == 0:
                                    state_map[(agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, 0)] = state_idx
                                    reverse_state_map[state_idx] = (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, 0)
                                    state_idx += 1

    n_states = state_idx

    print("Computing successor states and probabilities ... ")

    transition_probs = defaultdict(list)
    successor_states = defaultdict(list)

    if return_matrix:
        matrix = np.zeros((n_states, n_states, n_actions), dtype=np.float32)
    else:
        matrix = None

    for tup, state_idx in tqdm(state_map.items()):

        agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food = tup

        if (food_x is not None) and (food_y is not None) and (agent_x == food_x) and (agent_y == food_y):
            next_food = 0
        else:
            next_food = food

        next_ghost_y = int(np.clip(ghost_y + direction_map[ghost_direction][0], 0, nrow-1))
        next_ghost_x = int(np.clip(ghost_x + direction_map[ghost_direction][1], 0, ncol-1))

        next_loc_free = (standard_map[next_ghost_y, next_ghost_x] == 0)

        x = np.array([agent_y - ghost_y, agent_x - ghost_x], dtype=np.float32)
        norm = np.linalg.norm(x)
        if norm == 0.0:
            x_norm = x
        else:
            x_norm = x * (1/norm)
        prods = np.array([-np.inf for _ in range(n_actions)], dtype=np.float32)

        for ghost_act in range(n_actions):
            if next_loc_free and ghost_act == reverse_map[ghost_direction]:
                continue
            next_ghost_direction = ghost_act if ghost_act <= 3 else ghost_direction
            next_ghost_y = int(np.clip(ghost_y + direction_map[next_ghost_direction][0], 0, nrow-1))
            next_ghost_x = int(np.clip(ghost_x + direction_map[next_ghost_direction][1], 0, ncol-1))
            if standard_map[next_ghost_y, next_ghost_x] == 0:
                prods[ghost_act] = np.dot(x, np.array(direction_map[next_ghost_direction]))

        ghost_act_probs = np.zeros(n_actions, dtype=np.float32)
        available_ghost_acts = np.where(prods != -np.inf)[0]
        n = len(available_ghost_acts)
        ghost_act_probs[available_ghost_acts] = ghost_rand_prob / float(max(1, n-1))
        ghost_act_probs[np.argmax(prods)] = 1.0 - ghost_rand_prob if n > 1 else 1.0
        
        assert np.any(prods != np.array([-np.inf for _ in range(n_actions)]))
        assert np.sum(ghost_act_probs) == 1.0, f"sum: {np.sum(ghost_act_probs)}, probs: {ghost_act_probs}"

        # fill out the successor states dictionary

        for agent_act in range(n_actions):
            if next_loc_free and agent_act == reverse_map[agent_direction]:
                next_agent_direction = agent_direction
            else:
                next_agent_direction = agent_act if agent_act <= 3 else agent_direction
            next_agent_y = int(np.clip(agent_y + direction_map[next_agent_direction][0], 0, nrow-1))
            next_agent_x = int(np.clip(agent_x + direction_map[next_agent_direction][1], 0, ncol-1))
            if standard_map[next_agent_y, next_agent_x] == 1:
                next_agent_y = int(np.clip(agent_y + direction_map[agent_direction][0], 0, nrow-1))
                next_agent_x = int(np.clip(agent_x + direction_map[agent_direction][1], 0, ncol-1))
                if standard_map[next_agent_y, next_agent_x] == 1:
                    next_agent_y = agent_y
                    next_agent_x = agent_x
                next_agent_direction = agent_direction

            for ghost_act in available_ghost_acts:

                next_ghost_y = int(np.clip(ghost_y + action_map[ghost_act][0], 0, nrow-1))
                next_ghost_x = int(np.clip(ghost_x + action_map[ghost_act][1], 0, ncol-1))
                next_ghost_direction = ghost_act if ghost_act <= 3 else ghost_direction

                if (next_agent_y, next_agent_x) == (ghost_y, ghost_x):
                    successor_states[state_idx].append(state_map[(next_agent_y, next_agent_x, next_agent_direction, ghost_y, ghost_x, ghost_direction, next_food)])
                else:
                    successor_states[state_idx].append(state_map[(next_agent_y, next_agent_x, next_agent_direction, next_ghost_y, next_ghost_x, next_ghost_direction, next_food)])
                    
        # remove duplicates
        successor_states[state_idx] = list(set(successor_states[state_idx]))
        # sort the successor states by numerical order
        successor_states[state_idx].sort()

        # initialize the zero probability vector over the successor states for each action
        for a in range(n_actions):
            transition_probs[(state_idx, a)] = np.array([0.0 for _ in range(len(successor_states[state_idx]))], dtype=np.float32)

        for agent_act in range(n_actions):
            if next_loc_free and agent_act == reverse_map[agent_direction]:
                next_agent_direction = agent_direction
            else:
                next_agent_direction = agent_act if agent_act <= 3 else agent_direction
            next_agent_y = int(np.clip(agent_y + direction_map[next_agent_direction][0], 0, nrow-1))
            next_agent_x = int(np.clip(agent_x + direction_map[next_agent_direction][1], 0, ncol-1))
            if standard_map[next_agent_y, next_agent_x] == 1:
                next_agent_y = int(np.clip(agent_y + direction_map[agent_direction][0], 0, nrow-1))
                next_agent_x = int(np.clip(agent_x + direction_map[agent_direction][1], 0, ncol-1))
                if standard_map[next_agent_y, next_agent_x] == 1:
                    next_agent_y = agent_y
                    next_agent_x = agent_x
                next_agent_direction = agent_direction

            for ghost_act in available_ghost_acts:

                next_ghost_y = int(np.clip(ghost_y + action_map[ghost_act][0], 0, nrow-1))
                next_ghost_x = int(np.clip(ghost_x + action_map[ghost_act][1], 0, ncol-1))
                next_ghost_direction = ghost_act if ghost_act <= 3 else ghost_direction

                if (next_agent_y, next_agent_x) == (ghost_y, ghost_x):
                    idx = successor_states[state_idx].index(state_map[(next_agent_y, next_agent_x, next_agent_direction, ghost_y, ghost_x, ghost_direction, next_food)])
                    if return_matrix:
                        matrix[state_map[(next_agent_y, next_agent_x, next_agent_direction, ghost_y, ghost_x, ghost_direction, next_food)], state_idx, agent_act] += ghost_act_probs[ghost_act]
                else:
                    idx = successor_states[state_idx].index(state_map[(next_agent_y, next_agent_x, next_agent_direction, next_ghost_y, next_ghost_x, next_ghost_direction, next_food)])
                    if return_matrix:
                        matrix[state_map[(next_agent_y, next_agent_x, next_agent_direction, next_ghost_y, next_ghost_x, next_ghost_direction, next_food)], state_idx, agent_act] += ghost_act_probs[ghost_act]

                transition_probs[(state_idx, agent_act)][idx] += ghost_act_probs[ghost_act]

    return successor_states, transition_probs, matrix, n_states, state_map, reverse_state_map


def create_pacman_end_component(
    standard_map: np.array, 
    agent_x_term: int,
    agent_y_term: int,
    state_map: Dict[Tuple[int, int, int, int, int, int], int],
    n_directions: int = 4, 
    n_ghosts: int = 1, 
    food: bool = False,
):

    sec = []

    nrow = standard_map.shape[0]
    ncol = standard_map.shape[1]

    grid = np.arange(nrow*ncol).reshape(nrow, ncol)

    assert n_ghosts == 1, f"function only supports n_ghosts=1 not {n_ghosts}"

    direction_map = {0: (0, -1), # left
                    1: (0, 1), # right
                    2: (1, 0), # down
                    3: (-1, 0), # up
                }

    for ghost_y in range(nrow):
        for ghost_x in range(ncol):
            for ghost_direction in range(n_directions):

                infront_ghost_y = int(np.clip(ghost_y + direction_map[ghost_direction][0], 0, nrow-1))
                infront_ghost_x = int(np.clip(ghost_x + direction_map[ghost_direction][1], 0, ncol-1))

                behind_ghost_y = int(np.clip(ghost_y - direction_map[ghost_direction][0], 0, nrow-1))
                behind_ghost_x = int(np.clip(ghost_x - direction_map[ghost_direction][1], 0, ncol-1))

                if standard_map[infront_ghost_y, infront_ghost_x] == 1 and standard_map[behind_ghost_y, behind_ghost_x] == 1:
                    continue

                if not(standard_map[agent_y_term, agent_x_term] == 0 and standard_map[ghost_y, ghost_x] == 0):
                    continue

                if (agent_y_term, agent_x_term) == (ghost_y, ghost_x):
                    continue

                if food:
                    sec.append(state_map[(agent_y_term, agent_x_term, 2, ghost_y, ghost_x, ghost_direction, 1)])

                sec.append(state_map[(agent_y_term, agent_x_term, 2, ghost_y, ghost_x, ghost_direction, 0)])

    return sec