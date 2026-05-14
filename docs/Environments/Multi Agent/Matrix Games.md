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

For the standard MASA MARL wrapper stack, create them through `make_marl_env`.
This applies `LabelledParallelEnv -> ConstrainedMarkovGameEnv` and uses the
environment's default `label_fn` and `cost_fn` when available:

```python
from masa.common.constraints.multi_agent.cmg import Budget
from masa.common.utils import make_marl_env

env = make_marl_env(
    "chicken_matrix",
    "cmg",
    budgets=[Budget(amount=10.0, agents=("player_0", "player_1"), name="shared")],
)
```

Registered MARL environment ids are `bertrand_matrix`, `chicken_matrix`,
`congestion_matrix`, `dpgg_matrix`, and `inspection_matrix`.

`make_marl_env` also accepts `record_video=True`, `video_folder`, and
`video_kwargs` for renderable PettingZoo parallel environments. Video recording
requires `render()` to return RGB array frames. The current matrix games do not
yet implement concrete image renderers, so recording is available for renderable
PettingZoo environments and plugins until matrix-game renderers are added.

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
