# Matrix Games

MASA currently includes the following multi-agent matrix-game environments under `masa.envs.multiagent.matrix`.

All of these environments:

- use the PettingZoo `ParallelEnv` API,
- expose binary global observations,
- provide a `label_fn` and `cost_fn` for safety-aware training, and
- default to flattened observations.

You can instantiate them directly from their module paths, for example:

```python
from masa.envs.multiagent.matrix.chicken import ChickenMatrix

env = ChickenMatrix()
```

## Available Games

- [Bertrand](Matrix%20Games/Bertrand)
- [Chicken](Matrix%20Games/Chicken)
- [Congestion](Matrix%20Games/Congestion)
- [Dynamic Public Goods Game](Matrix%20Games/DPGG)
- [Inspection](Matrix%20Games/Inspection)

```{toctree}
:hidden:

Matrix Games/Bertrand
Matrix Games/Chicken
Matrix Games/Congestion
Matrix Games/DPGG
Matrix Games/Inspection
```
