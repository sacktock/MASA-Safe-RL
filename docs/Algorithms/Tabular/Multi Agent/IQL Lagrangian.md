# IQL Lagrangian

Source: `masa/algorithms/multi_agent/iql_lagrangian.py`

`IQLLagrangian` uses the same penalized reward as [IQL Lambda](IQL%20Lambda.md), but adapts one multiplier for each selected CMG budget. Multipliers remain fixed during an episode. After a batch of complete training episodes, each dual variable is updated by

$$
\lambda_b
\leftarrow
\Pi_{[0,\lambda_{\max}]}
\left(
\lambda_b + \eta(\overline C_b-B_b)
\right),
$$

where `B_b` is the budget cap and `C̄_b` is the batch mean of the budget's cumulative episode cost.

```python
from masa.algorithms.multi_agent import IQLLagrangian

model = IQLLagrangian(
    env,
    horizon=20,
    n_states=32,
    encode=binary_state,
    cost_lambda={"shared": 0.0},
    lambda_lr=0.02,
    dual_update_every=10,
    lambda_max=None,
    ql_kwargs={"gamma": 1.0, "exploration": "epsilon_greedy"},
)
```

A partial dual batch persists across repeated calls to `train`. Evaluation does not update Q-tables, exploration schedules, multipliers, or the partial dual batch. Episode rows record `lambda/<budget>` for the multiplier used in that rollout and `lambda_next/<budget>` after any episode-end update.

## Objective convention

CMG caps are undiscounted finite-episode sums. `IQLLagrangian` therefore defaults the underlying learners to `gamma=1.0` and rejects an explicitly supplied different value. This keeps the penalty objective and the dual accounting aligned.

## Interpretation

This is a compact independent-learning baseline. It tries to control **mean episode cost** through a simple projected dual ascent update. It does not guarantee that every realized trajectory satisfies its budget, nor does it supply a convergence guarantee for a general non-stationary Markov game.
