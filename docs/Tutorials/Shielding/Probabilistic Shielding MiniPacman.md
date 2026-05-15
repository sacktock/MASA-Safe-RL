# Probabilistic Shielding MiniPacman

This tutorial adapts `masa/examples/prob_shield_example.py` into a guided inspection of probabilistic shielding. Instead of running a long PPO training job, it builds the shielded MiniPacman environment and inspects what the wrapper adds.

Runnable notebook: [notebooks/tutorials/09_probabilistic_shielding_minipacman.ipynb](../../../notebooks/tutorials/09_probabilistic_shielding_minipacman.ipynb)

## Build MiniPacman with PCTL

MiniPacman is a discrete tabular environment with an exposed transition model. That makes it suitable for `ProbShieldWrapperDisc`, which needs known safety dynamics.

```python
from masa.common.utils import make_env
from masa.envs.tabular.mini_pacman import cost_fn, label_fn

env = make_env(
    "mini_pacman",
    "pctl",
    100,
    label_fn=label_fn,
    cost_fn=cost_fn,
    alpha=0.01,
)
```

The `pctl` constraint uses `cost_fn` to mark ghost collisions as unsafe and tracks whether the episode satisfies the probabilistic safety threshold.

## Add the Shield

The script example wraps that environment with `ProbShieldWrapperDisc`:

```python
from masa.prob_shield.prob_shield_wrapper_v1 import ProbShieldWrapperDisc

shielded_env = ProbShieldWrapperDisc(
    env,
    init_safety_bound=0.01,
    theta=1e-12,
    max_vi_steps=2_000,
    granularity=10,
)
```

During construction, the wrapper:

- builds `successor_states_matrix` and `probabilities`,
- computes interval safety bounds in `safety_lb`,
- records `max_successors`,
- replaces the observation space with a dictionary containing `orig_obs` and `safety_bound`,
- replaces the action space with a `MultiDiscrete` augmented action format.

## Inspect the Augmented Interface

After reset, the observation has two important pieces:

```python
obs, info = shielded_env.reset(seed=0)

obs["orig_obs"]       # original MiniPacman discrete state
obs["safety_bound"]  # remaining safety budget
```

The augmented action format is:

```text
[primary_action, fallback_action, beta_0, beta_1, ..., beta_K]
```

The first two entries select candidate MiniPacman actions. The remaining entries encode candidate successor safety-bound choices from `0` to `granularity`.

## Projection Geometry on a Toy MDP

The projection step follows the geometry described in the probabilistic shielding paper: action choices live in a probability simplex, and the safety budget cuts out the unsafe part of that simplex. The tiny MDP below uses exact reach-unsafe values so the clipping operation is visible before we inspect the larger MiniPacman state.

```{figure} ../../_static/tutorials/probabilistic_shielding_minipacman/simplex_projection.svg
:alt: Toy MDP and simplex projection for probabilistic shielding.
:width: 1040px

For budget `q=0.10`, `_project_act` keeps action distributions inside the green safe half-space `0.02*pi0 + 0.08*pi1 + 0.20*pi2 <= 0.10`. The red part of the simplex is the clipped-away region above that budget.
```

## Inspect Projected Safe Actions

The notebook uses `_parse_act` and `_project_act` to show how candidate augmented actions are projected. These are internal inspection hooks, not a stable public API.

The projection returns:

- `safe_actions`: a probability distribution over original MiniPacman actions,
- `bounds`: projected successor safety bounds,
- `proj_penalty`: how much the proposed bounds were changed,
- `margin_penalty`: a diagnostic margin term.

That inspection is the heart of the tutorial: it shows how the same MiniPacman action is filtered through the safety budget before the wrapped environment is stepped.

## Step Once

When the shielded environment steps:

1. the wrapper parses the augmented action,
2. projects it to a safe distribution over original actions,
3. samples an original MiniPacman action,
4. steps the PCTL-constrained environment,
5. updates the next observation's `safety_bound`.

The returned `info` still contains MASA constraint data:

```python
info["constraint"]
info["margin_penalty"]
info["proj_penalty"]
```

## Where Training Fits

The original example trains PPO on the shielded environment. This tutorial intentionally stops before training so it stays fast and readable. A full training run would pass one shielded env to PPO for training and a separate shielded env for evaluation.
