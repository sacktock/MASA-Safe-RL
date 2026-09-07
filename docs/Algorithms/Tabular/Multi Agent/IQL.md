# IQL

Source: `masa/algorithms/multi_agent/iql.py`

`IQL` implements independent tabular Q-learning by coordinating one existing MASA `QL` instance per PettingZoo agent. So, the class uses what is already available from the single-agent Q Learning implementation, including the Q-Learning target and exploration logic. Hence, its only new responsibility is to collect all active actions before calling `ParallelEnv.step(actions)` once, then feed one transition to each learner.

## Basic use

```python
from masa.algorithms.multi_agent import IQL
from masa.examples.cmg_iql import binary_state, make_chicken_cmg

env = make_chicken_cmg(horizon=20)
model = IQL(
    env,
    horizon=20,
    n_states=32,
    encode=binary_state,
    seed=0,
    ql_kwargs={
        "alpha": 0.1,
        "gamma": 1.0,
        "exploration": "epsilon_greedy",
        "epsilon_decay_frames": 16_000,
    },
)

train_rows = model.train(episodes=1000)
eval_rows = model.evaluate(episodes=100, deterministic=True)
```

`model.learners[agent]` is the underlying `QL` object and `model.q_tables[agent]` is its live Q-table.

## Observation encoding

Each Q-table needs an integer state. For a zero-based `Discrete` observation space, `n_states` and `encode` may be omitted. Other spaces require:

- `n_states`: the size of the encoded observation set;
- `encode(observation)`: a function returning an integer in `[0, n_states)`.

Both arguments may also be mappings keyed by agent for heterogeneous games.

`IQL` augments the encoded observation with elapsed decision time. For an observation id `s` and time `t`, the table index is `t * n_states + s`. This makes a configured finite-horizon game Markov even when the original observation does not contain time. The environment must end every episode by the supplied `horizon`.

## Independence semantics

All agents choose before any learner is updated. Each policy has a separate table, exploration schedule, and learner seed. A policy sees only the observation that its PettingZoo environment supplies. In `ChickenMatrix`, that observation is global by environment design; `IQL` does not add other agents' actions or Q-values.

## Metrics

`train` and `evaluate` return one flat dictionary per episode. Rows include:

- `return/<agent>` and `shaped_return/<agent>`;
- CMG episode metrics such as `<budget>_cum_cost` and `<budget>_satisfied`;
- `steps` and, for training rows, `episode`.

For plain `IQL`, shaped returns equal original returns and `model.lambdas` is empty.
