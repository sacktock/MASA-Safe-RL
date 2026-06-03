# FrozenLake Shielding

This tutorial-style note shows how FrozenLake can be used with MASA's existing discrete probabilistic shield. FrozenLake is a small discrete MDP with known transition probabilities, so shield synthesis can run directly over the environment's exact transition matrix.

## Build The Environment

```python
from masa.common.utils import make_env

env = make_env(
    "FrozenLake",
    "PCTL",
    100,
    env_kwargs={"is_slippery": False},
    constraint_kwargs={"alpha": 0.01},
)
```

The default FrozenLake safety property labels hole tiles as `"hole"` and assigns cost `1.0` to those labels. Start, frozen, and goal tiles are labelled as `"start"`, `"frozen"`, and `"goal"`.

## Add The Shield

```python
from masa.prob_shield.prob_shield_wrapper_v1 import ProbShieldWrapperDisc

shielded_env = ProbShieldWrapperDisc(
    env,
    init_safety_bound=1e-12,
    theta=1e-12,
    max_vi_steps=2_000,
    granularity=10,
)
```

During construction, `ProbShieldWrapperDisc` reads FrozenLake's transition matrix, computes reach-hole safety bounds with interval value iteration, and augments the Gymnasium interface with a remaining safety budget.

After reset:

```python
obs, info = shielded_env.reset(seed=0)

obs["orig_obs"]       # original FrozenLake state id
obs["safety_bound"]  # remaining probability budget for reaching a hole
```

The shielded action format is the same as other discrete probabilistic shields:

```text
[primary_action, fallback_action, beta_0, beta_1, ..., beta_K]
```

The wrapper projects the proposed action onto a safe distribution over FrozenLake's original actions before stepping the base environment.

Runnable example: `masa/examples/prob_shield_frozen_lake_example.py`.
