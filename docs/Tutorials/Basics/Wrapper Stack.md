# Wrapper Stack

This tutorial shows that `make_env(...)` is a convenience around a concrete wrapper stack. You will build the same `colour_grid_world` CMDP environment two ways:

- with `make_env(...)`,
- manually with each wrapper in order.

Runnable notebook: [notebooks/tutorials/03_wrapper_stack.ipynb](../../../notebooks/tutorials/03_wrapper_stack.ipynb)

## CPU-First Setup

Use the same CPU-first setup as the earlier tutorials:

```python
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
```

## Imports

The manual path uses the same pieces that `make_env` applies internally.

```python
from pprint import pprint

from masa.plugins.helpers import load_plugins
from masa.common.constraints.cmdp import CumulativeCostEnv
from masa.common.labelled_env import LabelledEnv
from masa.common.utils import make_env
from masa.common.wrappers import (
    ConstraintMonitor,
    RewardMonitor,
    TimeLimit,
    get_wrapped,
    is_wrapped,
)
from masa.envs.tabular.colour_grid_world import ColourGridWorld, cost_fn, label_fn

load_plugins()
```

## Build with `make_env`

`make_env` looks up the environment and constraint by registry name, constructs the base environment, and applies MASA's standard wrapper order.

```python
def build_factory_env():
    return make_env(
        "colour_grid_world",
        "cmdp",
        5,
        label_fn=label_fn,
        cost_fn=cost_fn,
        budget=0.0,
    )
```

## Build the Same Stack Manually

The equivalent manual stack is:

```text
ColourGridWorld
-> TimeLimit
-> LabelledEnv
-> CumulativeCostEnv
-> ConstraintMonitor
-> RewardMonitor
```

In code:

```python
def build_manual_env():
    env = ColourGridWorld()
    env = TimeLimit(env, 5)
    env = LabelledEnv(env, label_fn)
    env = CumulativeCostEnv(env, cost_fn=cost_fn, budget=0.0)
    env = ConstraintMonitor(env)
    env = RewardMonitor(env)
    return env
```

## Inspect the Wrapper Chain

`is_wrapped` answers whether a wrapper appears anywhere in the chain. `get_wrapped` returns the first matching wrapper object.

```python
WRAPPERS = (TimeLimit, LabelledEnv, CumulativeCostEnv, ConstraintMonitor, RewardMonitor)

def summarize_wrappers(env):
    return {
        wrapper.__name__: {
            "present": is_wrapped(env, wrapper),
            "found_type": type(get_wrapped(env, wrapper)).__name__,
        }
        for wrapper in WRAPPERS
    }

factory_env = build_factory_env()
manual_env = build_manual_env()

factory_summary = summarize_wrappers(factory_env)
manual_summary = summarize_wrappers(manual_env)

print("factory stack")
pprint(factory_summary)
print("manual stack")
pprint(manual_summary)

assert factory_summary == manual_summary

factory_env.close()
manual_env.close()
```

## Compare Behaviour

Both environments should emit the same observations, rewards, labels, constraint metrics, and done flags for the same seed and actions.

```python
ACTION_NAMES = {0: "left", 1: "right", 2: "down", 3: "up", 4: "stay"}

def rollout(build_env, actions, *, seed):
    env = build_env()
    obs, info = env.reset(seed=seed)
    rows = [
        {
            "event": "reset",
            "obs": int(obs),
            "labels": sorted(info["labels"]),
            "constraint": info["constraint"],
        }
    ]

    for step, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "event": f"step_{step}",
                "action": ACTION_NAMES[action],
                "obs": int(obs),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "labels": sorted(info["labels"]),
                "constraint": info["constraint"],
                "metrics": info.get("metrics"),
            }
        )
        if terminated or truncated:
            break

    env.close()
    return rows

actions = [2, 2, 2, 2]
factory_rows = rollout(build_factory_env, actions, seed=1)
manual_rows = rollout(build_manual_env, actions, seed=1)

pprint(factory_rows)
assert factory_rows == manual_rows
```

The final row reaches the blue state, so both environments should report:

- `labels == ["blue"]`,
- `constraint["step"]["cost"] == 1.0`,
- `constraint["step"]["violation"] == 1.0`.

## Why Order Matters

- `TimeLimit` comes first so truncation is part of the base interaction before safety monitoring.
- `LabelledEnv` must run before the constraint wrapper because constraints consume `info["labels"]`.
- `CumulativeCostEnv` updates the stateful safety monitor.
- `ConstraintMonitor` reads the constraint and writes `info["constraint"]`.
- `RewardMonitor` is last here so it can add reward and episode-length metrics without changing safety logic.

Most users should call `make_env`. Manual construction is useful when you need to understand, debug, or extend the stack.
