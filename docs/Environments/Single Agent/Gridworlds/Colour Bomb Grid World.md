# Colour Bomb Grid World

The Colour Bomb family extends the simple coloured gridworld idea with walls, multiple labelled regions, and bomb states.

MASA provides three variants:

- `colour_bomb_grid_world`
- `colour_bomb_grid_world_v2`
- `colour_bomb_grid_world_v3`

## Shared Gridworld Conventions

All three environments are single-agent tabular gridworlds with explicit stochastic transition models. They expose a full transition
matrix via `get_transition_matrix()`.

They use the standard gridworld action convention:

- `0`: move left
- `1`: move right
- `2`: move down
- `3`: move up
- `4`: stay in place

When slip is enabled, the intended action is taken with high probability and the remaining probability mass is spread uniformly over
the other actions.

## Shared Family Semantics

All three use:

- action space `Discrete(5)`,
- bomb labels as the default source of safety cost,
- `cost_fn(labels) = 1.0` when `"bomb"` is present and `0.0` otherwise.

## `colour_bomb_grid_world`

This is a `9 x 9` gridworld with walls, bombs, and several coloured regions.

- Observation space: `Discrete(81)`
- Slip probability: `0.1`
- Labels: `{"start"}`, `{"green"}`, `{"yellow"}`, `{"blue"}`, `{"pink"}`, `{"bomb"}`

The rewarding terminal states are the yellow, blue, and pink states. Reaching one of those states yields reward `1.0` and ends the
episode.

Bomb states are not terminal by themselves, and the green state is labelled but is not treated as a goal in this first variant.

## `colour_bomb_grid_world_v2`

This version scales the map up to `15 x 15` and introduces:

- five possible start states,
- five coloured goal groups: green, yellow, red, blue, and pink,
- bomb states,
- medic states,
- walls,
- slip probability `0.1`.

The interface is:

- observation space `Discrete(225)`,
- reward `1.0` on any coloured goal state,
- labels `{"start"}`, `{"green"}`, `{"yellow"}`, `{"red"}`, `{"blue"}`, `{"pink"}`, `{"bomb"}`, `{"medic"}`.

Two implementation details matter for algorithms:

1. The environment does not terminate on goal states.
2. After entering a goal state, the next-state distribution is replaced with a uniform reset over the predefined start states.

Medic states are passed to the transition helper as `safe_states`, which means slip is disabled there. In other words, medic states
behave like locally deterministic recovery points.

This makes `colour_bomb_grid_world_v2` a good fit for continuing-task formulations and temporal-logic constraints where the agent may
need to recover from unsafe prefixes and continue collecting reward.

## `colour_bomb_grid_world_v3`

This environment keeps the `15 x 15` geometry from `v2` but replicates it across five coloured zones, producing a state space of
`5 * 15 * 15 = 1125` discrete states.

The interface is:

- observation space `Discrete(1125)`,
- the same label set as `v2`.

The active zone is encoded in the state index. Each zone corresponds to one active colour:

- zone `0`: green
- zone `1`: yellow
- zone `2`: red
- zone `3`: blue
- zone `4`: pink

Reward is given only when the label on the current state matches the active colour for that zone.

As in `v2`, entering a goal-labelled state does not terminate the episode. Instead, future transitions from that state are replaced
with a uniform jump to one of the predefined start states.

This variant is particularly useful for multi-task or non-stationary-style specifications where the same geometry is reused under
different active objectives. Like `v2`, it is a continuing task rather than a terminating episodic one.
