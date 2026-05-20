# Continuous Safe RL Baselines

This page is a stub for the future continuous-action safe RL baselines tutorial. The current runnable baseline in this part of the codebase is `PPO`; `CPO` and `PPO Lagrangian` are important comparison points, but their docs pages are placeholders rather than tutorial-ready implementations.

Runnable notebook: [notebooks/tutorials/07_continuous_safe_rl_baselines.ipynb](../../../notebooks/tutorials/07_continuous_safe_rl_baselines.ipynb)

## Current Scope

| Baseline | Status | Tutorial state | Safety role |
| --- | --- | --- | --- |
| `PPO` | implemented | runnable scaffold only | uses MASA constraint wrappers for logging; no constrained objective in base PPO |
| `CPO` | mentioned / placeholder docs | not runnable here yet | future constrained policy optimization baseline |
| `PPO Lagrangian` | mentioned / placeholder docs | not runnable here yet | future Lagrangian penalty baseline for PPO-style training |

## Continuous Cartpole Scaffold

`cont_cartpole` is the tiny continuous-control environment for this tutorial family. It uses:

- observation space `Box(4,)`,
- action space `Box(1,)`,
- label `{"stable"}` while cart position and pole angle remain in bounds,
- cost `0.0` when stable and `1.0` otherwise.

The MASA wrapper setup is:

```python
from masa.common.utils import make_env
from masa.envs.continuous.cartpole import cost_fn, label_fn

env = make_env(
    "cont_cartpole",
    "cmdp",
    200,
    label_fn=label_fn,
    cost_fn=cost_fn,
    budget=0.0,
)
```

This produces the same `info["labels"]` and `info["constraint"]` structure used throughout the earlier tutorials.

## PPO Stub

The notebook includes a minimal PPO configuration sketch:

```python
PPO_STUB_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 32,
    "batch_size": 32,
    "n_epochs": 2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "device": "cpu",
}
```

It intentionally does not train. A full tutorial should add vectorization or normalization choices, an evaluation protocol, multiple seeds, and safety-specific comparisons.

## Future Comparison Surface

The eventual continuous safe RL baselines tutorial should compare:

- `PPO` as the unconstrained neural policy-gradient baseline with MASA safety logging,
- `CPO` as a constrained policy optimization baseline,
- `PPO Lagrangian` as a learned-penalty baseline.

Until those constrained on-policy baselines are implemented, this page remains a lightweight placeholder that verifies the continuous environment and records the intended tutorial shape.
