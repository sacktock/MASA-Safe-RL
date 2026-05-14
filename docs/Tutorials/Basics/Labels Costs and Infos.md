# Labels, Costs, and Infos

This tutorial slows down the MASA environment loop. Instead of training an agent, you will step through `colour_grid_world` manually and inspect:

- `obs`,
- `reward`,
- `info["labels"]`,
- cost values,
- `info["constraint"]`,
- `info["metrics"]`,
- `terminated`,
- `truncated`.

Runnable notebook: [notebooks/tutorials/02_labels_costs_and_infos.ipynb](../../../notebooks/tutorials/02_labels_costs_and_infos.ipynb)

## CPU-First Setup

Set these before importing MASA/JAX modules:

```python
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
```

This tutorial does not train an agent, but the same CPU-first convention keeps all tutorials portable.

## Labels and Costs

MASA separates environment observations from semantic safety signals:

```text
observation -> label_fn -> labels -> cost_fn -> scalar cost
```

For `colour_grid_world`, the blue state is the unsafe labelled state.

```python
from pprint import pprint

from masa.envs.tabular.colour_grid_world import (
    BLUE_STATE,
    GOAL_STATE,
    START_STATE,
    cost_fn,
    label_fn,
)

representative_states = {
    "start": START_STATE,
    "blue": BLUE_STATE,
    "goal": GOAL_STATE,
}

label_cost_table = []
for name, obs in representative_states.items():
    labels = label_fn(obs)
    label_cost_table.append(
        {
            "name": name,
            "obs": int(obs),
            "labels": sorted(labels),
            "cost": float(cost_fn(labels)),
        }
    )

pprint(label_cost_table)
```

The important convention is that labels describe what is true about the current observation, while the cost function decides which labels matter for a particular safety constraint.

## Build the Environment

Use a CMDP-style cumulative cost constraint with a zero budget. That makes a single blue-state visit immediately visible in the metrics.

```python
from masa.plugins.helpers import load_plugins
from masa.common.utils import make_env

load_plugins()

def build_colour_env(max_episode_steps=20, budget=0.0):
    return make_env(
        "colour_grid_world",
        "cmdp",
        max_episode_steps,
        label_fn=label_fn,
        cost_fn=cost_fn,
        budget=budget,
    )

env = build_colour_env()
obs, info = env.reset(seed=0)

print("reset obs:", obs)
print('info["labels"]:', info["labels"])
print('info["constraint"]:')
pprint(info["constraint"])
env.close()
```

At reset, the `LabelledEnv` wrapper has already populated `info["labels"]`, and `ConstraintMonitor` has populated the initial constraint step metrics.

## A Rollout Helper

This helper records the fields you should inspect when debugging a MASA environment. It stops when either `terminated` or `truncated` becomes true.

```python
ACTION_NAMES = {0: "left", 1: "right", 2: "down", 3: "up", 4: "stay"}

def run_rollout(actions, *, seed, max_episode_steps=20, budget=0.0):
    env = build_colour_env(max_episode_steps=max_episode_steps, budget=budget)
    obs, info = env.reset(seed=seed)
    rows = [
        {
            "event": "reset",
            "obs": int(obs),
            "labels": sorted(info["labels"]),
            "constraint_step": info["constraint"]["step"],
        }
    ]

    for step, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        row = {
            "event": f"step_{step}",
            "action": ACTION_NAMES[action],
            "obs": int(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "labels": sorted(info["labels"]),
            "constraint_step": info["constraint"]["step"],
            "constraint_episode": info["constraint"].get("episode"),
            "metric_step": info["metrics"]["step"],
            "metric_episode": info.get("metrics", {}).get("episode"),
        }
        rows.append(row)
        if terminated or truncated:
            break

    env.close()
    return rows
```

## Cost Rollout

With seed `1`, four `down` actions reach the blue state. The blue label has cost `1.0`.

```python
cost_rows = run_rollout([2, 2, 2, 2], seed=1, max_episode_steps=20, budget=0.0)
pprint(cost_rows)
```

On the final row, expect:

- `labels == ["blue"]`,
- `constraint_step["cost"] == 1.0`,
- `constraint_step["violation"] == 1.0`,
- `constraint_step["cum_cost"] == 1.0`.

## Termination Rollout

`terminated` means the environment task ended. With seed `4`, this scripted path reaches the goal state and receives reward `1.0`.

```python
termination_actions = [2] * 8 + [1] * 8
termination_rows = run_rollout(termination_actions, seed=4, max_episode_steps=40, budget=0.0)
pprint(termination_rows)
```

On the final row, expect:

- `labels == ["goal"]`,
- `reward == 1.0`,
- `terminated is True`,
- `truncated is False`,
- `metric_episode["ep_reward"] == 1.0`.

## Truncation Rollout

`truncated` means an external limit stopped the episode. Here the environment has `max_episode_steps=3`, so it truncates before reaching a terminal state.

```python
truncation_rows = run_rollout([1, 1, 1, 1], seed=0, max_episode_steps=3, budget=0.0)
pprint(truncation_rows)
```

On the final row, expect:

- `terminated is False`,
- `truncated is True`,
- `metric_episode["ep_length"] == 3`.

## What to Remember

- `obs` and `reward` remain the environment's task interface.
- `info["labels"]` is the semantic bridge from observations to safety logic.
- `cost_fn(info["labels"])` is what cost-based constraints consume.
- `info["constraint"]["step"]` is the per-step safety view.
- `info["constraint"]["episode"]` and `info["metrics"]["episode"]` summarize the episode.
- `terminated` is task completion/failure from the environment; `truncated` is an external cutoff such as a time limit.
