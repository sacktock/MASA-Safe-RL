# Bridge Crossing

MASA provides two bridge-style tabular gridworlds:

- `bridge_crossing`
- `bridge_crossing_v2`

Both environments share the same interface and reward-cost structure, but use different lava layouts.

## Shared Gridworld Conventions

Both environments are single-agent tabular gridworlds with explicit stochastic transition models. They expose a full transition
matrix via `get_transition_matrix()`.

They use the standard gridworld action convention:

- `0`: move left
- `1`: move right
- `2`: move down
- `3`: move up
- `4`: stay in place

When slip is enabled, the intended action is taken with high probability and the remaining probability mass is spread uniformly over
the other actions.

## Shared Mechanics

The two variants are `20 x 20` gridworlds with:

- a fixed start state in the lower-left corner,
- a goal region occupying the top seven rows,
- a lava region in the middle of the map,
- slip probability `0.04`.

They both use:

- observation space `Discrete(400)`,
- action space `Discrete(5)`,
- labels `{"start"}`, `{"goal"}`, and `{"lava"}`,
- cost `1.0` on `"lava"` and `0.0` otherwise.

Reward is sparse:

- `1.0` on goal states,
- `0.0` elsewhere.

Episodes terminate immediately when the agent reaches either a goal state or a lava state.

## `bridge_crossing`

This is the canonical narrow-bridge layout. The lava occupies the left eight columns and right nine columns of rows `8:12`,
leaving a three-cell-wide bridge through the middle.

## `bridge_crossing_v2`

This variant keeps the same interface and dynamics but changes the lava geometry. Here the lava fills most of rows `8:12` from
columns `2:16`, plus one extra cell on the lower-left edge of the hazard.

It is useful when you want a second bridge-style benchmark without changing the overall problem class.
