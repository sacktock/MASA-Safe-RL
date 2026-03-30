# Colour Grid World

`colour_grid_world` is a `9 x 9` single-agent tabular gridworld.

## Shared Gridworld Conventions

This environment is a single-agent tabular gridworld with an explicit stochastic transition model. It exposes a full transition
matrix via `get_transition_matrix()`.

It uses the standard gridworld action convention:

- `0`: move left
- `1`: move right
- `2`: move down
- `3`: move up
- `4`: stay in place

When slip is enabled, the intended action is taken with high probability and the remaining probability mass is spread uniformly over
the other actions.

## Environment Details

- start state `0`,
- goal state `80`,
- slip probability `0.1`,
- one special blue state, one green state, and one purple state.

The environment uses:

- observation space `Discrete(81)`,
- action space `Discrete(5)`,
- labels `{"start"}`, `{"goal"}`, `{"blue"}`, `{"green"}`, and `{"purple"}`,
- cost `1.0` on `"blue"` and `0.0` otherwise.

Reward is sparse:

- `1.0` when the agent reaches the goal state,
- `0.0` otherwise.

The episode terminates when the goal state is reached.

This is a small benchmark for experiments where the reward target and the safety-relevant state are intentionally different.
