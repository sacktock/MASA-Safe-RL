# Markov Stag Hunt

`MarkovStagHunt` is a fully observable PettingZoo `ParallelEnv` in which agents repeatedly choose between safe individual foraging and risky,
high-value cooperation.

```python
from masa.envs.multiagent.tabular import MarkovStagHunt

env = MarkovStagHunt(render_mode="rgb_array")
observations, infos = env.reset(seed=0)
```

The default environment uses a `5 x 5` grid with two agents, two plants, one stag, and a 500-step horizon. Agents may occupy the same cell. Plants and successfully hunted stags respawn on unoccupied cells.

## Actions

- `0`: move left
- `1`: move right
- `2`: move up
- `3`: move down
- `4`: stay in place

All agents move simultaneously. Movement is clipped at the grid boundary. After movement, every plant and stag interaction is resolved from the complete set of new agent positions.

## Rewards and termination

The default rewards are:

- plant harvest: `+2` for each agent on the plant cell,
- successful stag hunt: `+10` for each participating hunter,
- failed stag hunt: `-2` for each mauled agent.

A stag hunt succeeds when at least `min_hunters_for_stag` agents enter its cell on the same step. With the default value of two, entering alone causes a mauling and immediately terminates the episode for every agent. The episode otherwise truncates at `max_moves`.

Each surviving stag independently moves one cell toward the nearest agent with probability `stag_move_prob`, which defaults to `0.2`.

## Observations

Observations are global binary grid layers with one channel per player, one plant channel, one stag channel, and a final `self_mauled` channel. They are flattened by default. With the default configuration the shape is `(125,)`, corresponding to `5 x 5 x 5`; set `flatten_observations=False` to receive the spatial shape `(5, 5, 5)`.

The world layers are shared, but `self_mauled` is agent-specific. This lets the standard MASA labelling wrapper charge only the agent that attempted an unsafe solo hunt.

## Labels and cost

The default `label_fn` returns `{"safe"}` normally. On the terminal transition for a mauled agent it returns `{"mauled", "unsafe"}`. The default binary cost is therefore:

```python
cost = 1.0 if "unsafe" in labels else 0.0
```

Rendering supports `ansi`, `rgb_array`, and `human`. The human renderer uses a resizable, high-contrast meadow board with distinct player, plant, and stag markers. A simultaneous-action playable example is available in `notebooks/envs/multiagent/play_markov_stag_hunt.ipynb`.
