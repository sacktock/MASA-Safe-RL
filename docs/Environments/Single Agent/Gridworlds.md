# Gridworlds

MASA includes several single-agent tabular gridworlds. They all use discrete states, discrete actions, and explicit stochastic
transition models. Every environment on this page exposes a full transition matrix via `get_transition_matrix()`.

## Shared Action Convention

The gridworld helpers use the same five actions throughout:

- `0`: move left
- `1`: move right
- `2`: move down
- `3`: move up
- `4`: stay in place

When slip is enabled, the intended action is taken with high probability and the remaining probability mass is spread uniformly over
the other actions.

## Bridge Crossing

The `bridge_crossing` and `bridge_crossing_v2` environments are `20 x 20` gridworlds with:

- a fixed start state in the lower-left corner,
- a goal region occupying the top seven rows,
- a lava region in the middle of the map,
- slip probability `0.04`.

Both variants use:

- observation space `Discrete(400)`,
- action space `Discrete(5)`,
- labels `{"start"}`, `{"goal"}`, and `{"lava"}`,
- cost `1.0` on `"lava"` and `0.0` otherwise.

The reward is `1.0` on goal states and `0.0` elsewhere. Episodes terminate immediately when the agent reaches either a goal state or
a lava state.

### `bridge_crossing`

This is the canonical narrow-bridge layout. The lava occupies the left eight columns and right nine columns of rows `8:12`,
leaving a three-cell-wide bridge through the middle.

### `bridge_crossing_v2`

This variant keeps the same interface and dynamics but changes the lava geometry. Here the lava fills most of rows `8:12` from
columns `2:16`, plus one extra cell on the lower-left edge of the hazard. It is useful when you want a second bridge-style
benchmark without changing the overall problem class.

## Colour Grid World

`colour_grid_world` is a `9 x 9` tabular gridworld with:

- start state `0`,
- goal state `80`,
- slip probability `0.1`,
- one special blue state, one green state, and one purple state.

The environment uses:

- observation space `Discrete(81)`,
- action space `Discrete(5)`,
- labels `{"start"}`, `{"goal"}`, `{"blue"}`, `{"green"}`, and `{"purple"}`,
- cost `1.0` on `"blue"` and `0.0` otherwise.

The reward is sparse: the agent receives `1.0` only when it reaches the goal state. The episode then terminates.

This is a small benchmark for experiments where the reward target and the safety-relevant state are intentionally different.

## Colour Bomb Grid World

The Colour Bomb family extends the simple coloured gridworld idea with walls, multiple labelled regions, and bomb states.

### `colour_bomb_grid_world`

This is a `9 x 9` gridworld with walls, bombs, and several coloured regions.

- Observation space: `Discrete(81)`
- Action space: `Discrete(5)`
- Slip probability: `0.1`
- Labels: `{"start"}`, `{"green"}`, `{"yellow"}`, `{"blue"}`, `{"pink"}`, `{"bomb"}`
- Cost: `1.0` on `"bomb"`

The rewarding terminal states are the yellow, blue, and pink states. Reaching one of those states yields reward `1.0` and ends the
episode. Bomb states are not terminal by themselves, but they are the default source of safety cost.

Notably, the green state is labelled but is not treated as a goal in this first variant.

### `colour_bomb_grid_world_v2`

This version scales the map up to `15 x 15` and introduces:

- five possible start states,
- five coloured goal groups: green, yellow, red, blue, and pink,
- bomb states,
- medic states,
- walls,
- slip probability `0.1`.

The interface is:

- observation space `Discrete(225)`,
- action space `Discrete(5)`,
- reward `1.0` on any coloured goal state,
- cost `1.0` on `"bomb"`.

Two implementation details matter for algorithms:

1. The environment does **not** terminate on goal states.
2. After entering a goal state, the next-state distribution is replaced with a uniform reset over the predefined start states.

Medic states are passed to the transition helper as `safe_states`, which means slip is disabled there. In other words, medic states
behave like locally deterministic recovery points.

This makes `colour_bomb_grid_world_v2` a good fit for continuing-task formulations and temporal-logic constraints where the agent may
need to recover from unsafe prefixes and continue collecting reward.

### `colour_bomb_grid_world_v3`

This environment keeps the `15 x 15` geometry from `v2` but replicates it across **five coloured zones**, producing a state space of
`5 * 15 * 15 = 1125` discrete states.

The interface is:

- observation space `Discrete(1125)`,
- action space `Discrete(5)`,
- the same label set as `v2`,
- cost `1.0` on `"bomb"`.

The active zone is encoded in the state index. Each zone corresponds to one active colour:

- zone `0`: green
- zone `1`: yellow
- zone `2`: red
- zone `3`: blue
- zone `4`: pink

Reward is given only when the label on the current state matches the active colour for that zone. In code, this is the condition

```python
active_colour_dict[state // (grid_size**2)] in label_fn(state)
```

As in `v2`, entering a goal-labelled state does not terminate the episode. Instead, future transitions from that state are replaced
with a uniform jump to one of the predefined start states.

This variant is particularly useful for multi-task or non-stationary-style specifications where the same geometry is reused under
different active objectives. Like `v2`, it is a continuing task rather than a terminating episodic one.
