# Safety Gridworlds

MASA includes three single-agent environments adapted from the AI Safety Gridworlds style of benchmarks:

- `island_navigation`
- `conveyor_belt`
- `sokoban`

All three use the Gymnasium single-agent API with:

- discrete observations,
- `Discrete(4)` actions,
- default `label_fn` and `cost_fn` helpers for MASA constraints,
- `rgb_array` and `human` rendering modes.

These environments are useful when reward and safety come apart on purpose. The agent can often improve return by approaching or
triggering unsafe situations, so they are natural testbeds for CMDP budgets, reach-avoid constraints, and automaton-based safety
specifications.

## Shared Action Convention

The three Safety Gridworld ports use the same four movement actions:

- `0`: move right
- `1`: move up
- `2`: move left
- `3`: move down

If a move would cross a wall or boundary, the blocked object stays in place.

## Practical Use

All three environments are registered in MASA and can be created through `make_env`:

```python
from masa.common.utils import make_env
from masa.envs.discrete.island_navigation import label_fn, cost_fn

env = make_env(
    "island_navigation",
    "cmdp",
    300,
    label_fn=label_fn,
    cost_fn=cost_fn,
    budget=0.0,
)
```

To inspect the visual layouts interactively, see `notebooks/envs/play_safety_gridworlds.ipynb`.

```{toctree}
:hidden:

Safety Gridworlds/Island Navigation
Safety Gridworlds/Conveyor Belt
Safety Gridworlds/Sokoban
```
