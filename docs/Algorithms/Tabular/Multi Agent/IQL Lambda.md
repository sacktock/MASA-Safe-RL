# IQL Lambda

Source: `masa/algorithms/multi_agent/iql_lambda.py`

`IQLLambda` extends [IQL](IQL.md) with fixed, nonnegative penalties derived from CMG budget metrics. It still uses one ordinary MASA `QL` learner per agent.

For budget `b` over agents `G_b`, the CMG monitor reports the aggregate step cost

$$
c_{b,t}=\sum_{j\in G_b}c_{j,t}.
$$

Every member agent learns from

$$
\widetilde r_{i,t}
=
r_{i,t}-\sum_{b:i\in G_b}\lambda_b c_{b,t}.
$$

The Q-learning update receives the already-shaped reward, so a cost is not applied a second time and does not automatically terminate bootstrapping.

## Select budgets explicitly

```python
from masa.algorithms.multi_agent import IQLLambda

model = IQLLambda(
    env,
    horizon=20,
    n_states=32,
    encode=binary_state,
    cost_lambda={"shared": 1.0},
    ql_kwargs={"gamma": 1.0, "exploration": "epsilon_greedy"},
)
```

Prefer a **mapping** when you want precise control over which budgets affect learning. For example, `{"shared": 1.0}` penalizes only the shared budget, while other budgets are still monitored but don't affect the reward. A scalar applies the same multiplier to every budget. If selected budgets overlap, their penalties are added.

For the Chicken tutorial, one crash gives cost `1` to each agent and `2` to the shared budget. A multiplier of `1.0` on `shared` therefore subtracts `2.0` from each agent's reward on that step. This is the CMG budget's sum semantics, not an average over members.
