# Tabular Safe RL Baselines

This tutorial compares MASA's tabular safe RL baselines on one constrained tabular environment. The point is to read the reward/safety tradeoff each method encodes, not to rank algorithms from a tiny run.

Runnable notebook: [notebooks/tutorials/06_tabular_safe_rl_baselines.ipynb](../../../notebooks/tutorials/06_tabular_safe_rl_baselines.ipynb)

## Shared Setup

Use CPU-first setup before importing MASA/JAX modules:

```python
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
```

All algorithms use the same constrained environment:

```python
from masa.common.utils import make_env
from masa.envs.tabular.colour_grid_world import cost_fn, label_fn

def make_train_env():
    return make_env(
        "colour_grid_world",
        "cmdp",
        40,
        label_fn=label_fn,
        cost_fn=cost_fn,
        budget=0.0,
    )
```

`colour_grid_world` is small enough for tutorial execution, has discrete states/actions, and separates sparse reward from safety cost:

- `goal` gives task reward,
- `blue` gives safety cost,
- the CMDP budget is `0.0`.

## Probe the Signals

Before training, run scripted rollouts to inspect the data that algorithms receive:

- unsafe script: seed `1`, actions `[2, 2, 2, 2]`, reaches `blue`,
- goal script: seed `4`, actions `[2] * 8 + [1] * 8`, reaches `goal`.

The unsafe script shows `cost=1.0`, `violation=1.0`, `cum_cost > 0`, and `satisfied=0.0`. The goal script shows reward without violating the CMDP budget.

## Algorithms Compared

| Algorithm | Class | Safety mechanism |
| --- | --- | --- |
| `q_learning` | `QL` | Task `Q` table only; costs are logged but not penalized in the update target. |
| `q_learning_lambda` | `QL_Lambda` | Subtracts `cost_lambda * cost` from the reward target. |
| `lcrl` | `LCRL` | Maps violations to an absorbing-style value based on `r_min / (1 - gamma)`. |
| `sem` | `SEM` | Learns task `Q` plus auxiliary `D` and `C` safety tables that alter action selection. |
| `recreg` | `RECREG` | Learns task `Q`, backup `B`, risk `S`, and can report `override_rate` when actions are replaced. |

The tutorial uses this compact registry:

```python
ALGORITHMS = {
    "q_learning": (QL, {}),
    "q_learning_lambda": (QL_Lambda, {"cost_lambda": 1.0}),
    "lcrl": (LCRL, {"r_min": -1.0}),
    "sem": (SEM, {"r_min": -1.0, "cost_coef": 1.0}),
    "recreg": (
        RECREG,
        {
            "mode": "model_free",
            "model_checking": "none",
            "horizon": 3,
            "safety_prob": 0.2,
            "cost_coef": 2.0,
        },
    ),
}
```

## Tiny Smoke Runs

Each algorithm is trained with the same tiny configuration:

```python
algo.train(
    num_frames=20,
    eval_freq=10,
    log_freq=10,
    num_eval_episodes=1,
    stats_window_size=5,
)
```

These runs are intentionally too small for performance claims. They are diagnostics that prove:

- all five algorithms can train on the same MASA wrapper stack,
- train/eval logging includes reward and constraint metrics,
- the learned objects expose the expected state tables.

## Reading the Learned State

After training, inspect the object shapes:

- `QL`, `QL_Lambda`, and `LCRL` expose `Q`,
- `SEM` exposes `Q`, `D`, and `C`,
- `RECREG` exposes `Q`, backup table `B`, and risk table `S`.

Use the printed `train/rollout`, `train/stats`, and `eval/rollout` blocks to interpret reward and safety together. For example, `RECREG` may add `override_rate` to episode metrics when it replaces a risky task action with a backup action.

## What to Take Away

The algorithms differ less in the environment interaction loop than in how they turn cost and violation signals into learning or action-selection pressure:

- `QL` is the task-only baseline,
- `QL_Lambda` is a soft penalty baseline,
- `LCRL` makes violations strongly unattractive,
- `SEM` separates task and safety estimates,
- `RECREG` introduces backup-policy intervention.

For real comparisons, increase training frames, run multiple seeds, and summarize confidence intervals. Keep this tutorial run as a quick wiring and interpretation check.
