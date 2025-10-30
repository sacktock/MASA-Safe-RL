# Turn-Based (AEC) Multi Agent Environments
PettingZoo supports two types of environment interfaces:
- `Parallel`, for environments where at each timestep, each agent selects an action independently. Formally, such environments are representable as a **Partially Observable Stochastic Game**.
- `AEC`, for turn-based environments using the *Agent Environment Cycle* model. Parallel environments can be converted to AEC environments.

