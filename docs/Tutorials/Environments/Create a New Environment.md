# Create a New Environment

This tutorial shows the smallest useful path from a raw Gymnasium environment to a MASA-ready constrained environment.

Runnable notebook: [notebooks/tutorials/08_create_a_new_environment.ipynb](../../../notebooks/tutorials/08_create_a_new_environment.ipynb)

## Learning Path

You will build a deterministic 2x2 delivery task:

| State | Meaning | Label | Cost |
| --- | --- | --- | --- |
| `0` | start | `start` | `0.0` |
| `1` | spill | `spill` | `1.0` |
| `2` | safe lane | none | `0.0` |
| `3` | goal | `goal` | `0.0` |

The raw environment is a normal Gymnasium `Env` with:

- `observation_space = spaces.Discrete(4)`,
- `action_space = spaces.Discrete(4)`,
- reward `1.0` when the agent reaches the goal,
- `terminated=True` at the goal,
- no built-in safety logic.

## Labels and Costs

MASA keeps semantic information outside the raw observation and reward. The labelling function maps observations to atomic propositions:

```python
def label_fn(obs):
    labels = set()
    if obs == 0:
        labels.add("start")
    if obs == 1:
        labels.add("spill")
    if obs == 3:
        labels.add("goal")
    return labels
```

The CMDP cost function maps labels to cost:

```python
def cost_fn(labels):
    return 1.0 if "spill" in labels else 0.0
```

## MASA Registration

`make_env` looks up environments through MASA's registry, not Gymnasium's global registry. For a notebook-only environment, register the class directly and guard the registration so the cell can be rerun:

```python
from masa.common.registry import ENV_REGISTRY
from masa.common.utils import make_env

ENV_ID = "tutorial_tiny_delivery"

if ENV_ID not in ENV_REGISTRY.keys():
    ENV_REGISTRY.register(ENV_ID, TinyDeliveryEnv)
```

Then build the wrapped environment:

```python
env = make_env(
    ENV_ID,
    "cmdp",
    4,
    label_fn=label_fn,
    cost_fn=cost_fn,
    budget=0.0,
)
```

The raw observation remains the same, but `info` now contains:

- `info["labels"]` from `LabelledEnv`,
- `info["constraint"]["step"]` from the CMDP constraint,
- episode-level constraint metrics when the rollout ends,
- reward metrics from `RewardMonitor`.

## Expected Rollouts

The safe route goes down then right:

| actions | final obs | terminated | truncated | cum cost | satisfied |
| --- | --- | --- | --- | --- | --- |
| `[1, 0]` | `3` | `True` | `False` | `0.0` | `1.0` |

The unsafe route goes right through the spill, then down to the same goal:

| actions | spill step cost | spill violation | final obs | cum cost | satisfied |
| --- | --- | --- | --- | --- | --- |
| `[0, 1]` | `1.0` | `1.0` | `3` | `1.0` | `0.0` |

The truncation route stays at the start until `TimeLimit` ends the episode:

| actions | max episode steps | terminated | truncated | cum cost |
| --- | --- | --- | --- | --- |
| `[2, 2, 2]` | `3` | `False` | `True` | `0.0` |

## Promoting the Example

For a real MASA environment, move the environment class and helper functions into `masa/envs/...`, add a permanent `ENV_REGISTRY.register(...)` entry in the supported plugins module, and move the notebook assertions into `tests/`.

The core pattern stays the same: implement the Gymnasium API first, then define labels and costs, then use MASA's registry and `make_env` wrapper stack.
