# Multi Agent
For Multi Agent environments, we use the **PettingZoo** API for environment interfaces instead of Gymnasium (since it only targets the single-agent setting). Note that PettingZoo supports two types of environment interfaces:
- `Parallel`, for environments where at each timestep, each agent selects an action independently. Formally, such environments are representable as a **Partially Observable Stochastic Game**.
- `AEC`, for turn-based environments using the *Agent Environment Cycle* model. Parallel environments can be converted to AEC environments.

`Parallel` environments subsume AEC environments, since AEC environments can be seen as a wrapper over Parallel environments (hence the ease in convertibility). At the current moment, we only support `Parallel` environments.

```{toctree}
:hidden:

Multi Agent/Gridworlds
Multi Agent/Matrix Games
```
