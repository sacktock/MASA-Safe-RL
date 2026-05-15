# Safety Abstractions Pacman Coins

This tutorial explains why safety abstractions are a major efficiency tool for structured Pacman coin observations.

Runnable notebook: [notebooks/tutorials/10_safety_abstractions_pacman_coins.ipynb](../../../notebooks/tutorials/10_safety_abstractions_pacman_coins.ipynb)

## Why This Matters

`mini_pacman_with_coins` returns a structured observation tensor:

```text
(7, 10, 9)
```

The channels contain:

- coin locations,
- agent direction channels,
- ghost direction channels.

The full task observation needs coin locations because reward depends on collecting coins. Ghost-collision safety, however, only depends on the agent and ghost state. A safety abstraction lets the shield ignore reward-only details while keeping the original observation available to the learning algorithm.

For the mini environment, a naive exact task state model would need the 4,624 agent/ghost safety states plus 70 coin bits:

| Model | State count | What it remembers |
| --- | --- | --- |
| Naive original task state upper bound | `4,624 * 2^70 ~= 5.459e24` | agent/ghost safety state plus every coin mask |
| Safety abstract states | `4,624` | agent and ghost positions/directions only |

That is a `2^70 ~= 1.181e21`-fold reduction for safety analysis. This is the point of the tutorial.

The larger `pacman_with_coins` environment uses the same pattern, but `mini_pacman_with_coins` is faster for a runnable tutorial.

## Base Environment

The constrained task is built in the usual MASA way:

```python
from masa.common.utils import make_env
from masa.envs.discrete import mini_pacman_with_coins as coins

env = make_env(
    "mini_pacman_with_coins",
    "pctl",
    100,
    label_fn=coins.label_fn,
    cost_fn=coins.cost_fn,
    alpha=0.01,
)
```

The task-level `label_fn` reads the structured observation and marks `"ghost"` when Pacman collides with the ghost. The `cost_fn` maps that label to unsafe cost.

## Safety Abstraction

The environment module provides:

```python
coins.safety_abstraction(obs)
coins.abstr_label_fn(abstract_state)
coins.cost_fn(labels)
```

The abstraction extracts the agent and ghost state from the structured observation and maps it to a compact discrete state id. It ignores coin locations because coins affect reward, not ghost-collision safety.

The notebook demonstrates this by changing only one coin-channel value: the raw observation changes, but `safety_abstraction(obs)` stays the same.

## Shielded Environment

With the abstraction supplied, the shield can compute safety bounds:

```python
shielded_env = ProbShieldWrapperDisc(
    env,
    label_fn=coins.abstr_label_fn,
    cost_fn=coins.cost_fn,
    safety_abstraction=coins.safety_abstraction,
    init_safety_bound=0.01,
    theta=1e-12,
    max_vi_steps=2_000,
    granularity=10,
)
```

The shielded observation contains:

- `orig_obs`: the original structured `(7, 10, 9)` task observation,
- `safety_bound`: the current remaining safety budget.

The shield computes over abstract safety states through `safety_lb`, `successor_states_matrix`, and `probabilities`, while the learning algorithm can still observe the full coin task.

## What To Inspect

The tutorial inspects:

- the raw observation channels,
- the original-state versus abstract-state count,
- the abstract state and abstract labels,
- coin-channel invariance,
- shield safety bounds and successor dynamics,
- one projected candidate action,
- one shielded step with reward, PCTL info, `proj_penalty`, `margin_penalty`, and updated `safety_bound`.
