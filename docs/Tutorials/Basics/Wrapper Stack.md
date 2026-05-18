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
from pathlib import Path
from pprint import pprint

from gymnasium.wrappers import RecordVideo

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

## Record the Finished Stack

`RecordVideo` is not part of the semantic MASA stack. When `record_video=True`, `make_env` wraps the completed stack with Gymnasium's video recorder, so labels, constraints, and monitors behave the same while frames are saved from `render()`.

In the notebook, `Path(mkdtemp(prefix="masa-wrapper-stack-video-"))` creates a fresh directory in Python's default temporary location. On Linux this is usually `/tmp`, so the actual directory will look like `/tmp/masa-wrapper-stack-video-...`. The cell prints the exact directory and MP4 paths. Use a project-relative path, as below, if you want to keep the videos near the repo checkout.

`record_video_episode_trigger` is Gymnasium's `episode_trigger`. It receives the zero-based episode id and records that episode when it returns `True`. Common schedules are:

```python
record_every_episode = lambda episode_id: True
record_every_5_episodes_from_zero = lambda episode_id: episode_id % 5 == 0
record_human_episodes_5_10_15 = lambda episode_id: (episode_id + 1) % 5 == 0
```

For step-based schedules, pass Gymnasium's `step_trigger` through `video_kwargs`. The step id is global across episodes. This starts a fixed-length recording every 500 environment steps:

```python
video_kwargs={
    "step_trigger": lambda step_id: step_id > 0 and step_id % 500 == 0,
    "video_length": 500,
}
```

```python
video_dir = Path("videos/tutorial_wrapper_stack")
print("video directory", video_dir)

video_env = make_env(
    "colour_grid_world",
    "cmdp",
    len(actions),
    label_fn=label_fn,
    cost_fn=cost_fn,
    budget=0.0,
    env_kwargs={
        "render_mode": "rgb_array",
        "render_window_size": 96,
    },
    record_video=True,
    record_video_episode_trigger=lambda episode_id: True,
    video_folder=str(video_dir),
)

assert isinstance(video_env, RecordVideo)

try:
    video_env.reset(seed=2)
    for action in actions:
        _, _, terminated, truncated, _ = video_env.step(action)
        if terminated or truncated:
            break
finally:
    video_env.close()

recorded_videos = sorted(video_dir.glob("*.mp4"))
for path in recorded_videos:
    print(path)

assert recorded_videos
```

## Why Order Matters

- `TimeLimit` comes first so truncation is part of the base interaction before safety monitoring.
- `LabelledEnv` must run before the constraint wrapper because constraints consume `info["labels"]`.
- `CumulativeCostEnv` updates the stateful safety monitor.
- `ConstraintMonitor` reads the constraint and writes `info["constraint"]`.
- `RewardMonitor` is last here so it can add reward and episode-length metrics without changing safety logic.
- When enabled, `RecordVideo` sits outside the completed stack and observes rendered frames without changing MASA metadata.

Most users should call `make_env`. Manual construction is useful when you need to understand, debug, or extend the stack.
