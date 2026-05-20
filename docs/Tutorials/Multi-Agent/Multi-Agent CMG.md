# Multi-Agent CMG

This tutorial introduces MASA's multi-agent constrained Markov game path with a small repeated Chicken matrix game.

Runnable notebook: [notebooks/tutorials/12_multi_agent_cmg.ipynb](../../../notebooks/tutorials/12_multi_agent_cmg.ipynb)

## PettingZoo Parallel Shape

MASA's multi-agent environments use the PettingZoo `ParallelEnv` API. Instead of one observation, one reward, and one info dict, a parallel environment returns dictionaries keyed by agent id:

```python
obs, infos = env.reset(seed=0)
obs, rewards, terminations, truncations, infos = env.step(
    {"player_0": action_0, "player_1": action_1}
)
```

For `ChickenMatrix`, the agents are `player_0` and `player_1`. Each agent chooses one of:

```text
Swerve   = 0
Straight = 1
```

The raw observation records the previous joint action and whether the previous round was a crash.

## Chicken Safety Semantics

In Chicken, the unsafe outcome is:

```text
player_0: Straight
player_1: Straight
```

That joint action causes a crash. The environment's `label_fn` emits labels including:

```text
crash
unsafe
straight_straight
```

The default `cost_fn` maps any label set containing `unsafe` to cost `1.0`; otherwise the cost is `0.0`.

## MASA CMG Wrapper

The standard MASA multi-agent wrapper path is built through `make_marl_env`:

```python
from masa.common.constraints.multi_agent.cmg import Budget
from masa.common.utils import make_marl_env

env = make_marl_env(
    "chicken_matrix",
    "cmg",
    env_kwargs={"max_moves": 3},
    budgets=[
        Budget(amount=1.0, agents=("player_0",), name="player_0_budget"),
        Budget(amount=1.0, agents=("player_1",), name="player_1_budget"),
        Budget(amount=1.5, agents=("player_0", "player_1"), name="shared"),
    ],
)
```

This applies:

```text
ChickenMatrix -> LabelledParallelEnv -> ConstrainedMarkovGameEnv
```

After wrapping, labels are available as `infos[agent]["labels"]`, and CMG metrics are available from:

```python
env.constraint_step_metrics()
env.constraint_episode_metrics()
```

## Safe Rollout

A safe script such as `Swerve/Swerve` followed by `Straight/Swerve` produces rewards without crash labels.

The important metrics stay at zero:

```text
player_0_cost      = 0.0
player_1_cost      = 0.0
shared_cum_cost    = 0.0
shared_satisfied   = 1.0
satisfied          = 1.0
```

## Shared-Budget Rollout

Now consider one unsafe round:

```text
player_0: Straight
player_1: Straight
```

Both agents get the unsafe crash label, so each agent contributes cost `1.0`:

```text
player_0_cost = 1.0
player_1_cost = 1.0
shared_cost   = 2.0
```

The individual budgets each have amount `1.0`, so they still pass:

```text
player_0_budget_satisfied = 1.0
player_1_budget_satisfied = 1.0
```

The shared budget has amount `1.5`, so the combined cost fails:

```text
shared_cum_cost  = 2.0
shared_satisfied = 0.0
satisfied        = 0.0
```

That is the central CMG lesson: per-agent labels become per-agent costs, and budgets can aggregate those costs over any subset of agents. A rollout can be acceptable for each agent's individual budget while failing a shared safety constraint.
