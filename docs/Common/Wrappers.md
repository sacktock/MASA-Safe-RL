# Wrappers

Environment wrappers for MASA-Safe-RL.

This module contains small, composable `gymnasium.Wrapper` utilities that
(1) preserve access to constraint-related objects through wrapper chains,
(2) inject monitoring/metrics into `info`, (3) apply potential-based reward
shaping for DFA-based constraints, and (4) provide basic observation/reward
normalization and light-weight vector-environment helpers.

### Key conventions
* Constraint-enabled environments expose a `_constraint` object and (often)
  `label_fn` / `cost_fn` attributes. See [`masa.common.constraints.base.BaseConstraintEnv`][masa.common.constraints.base.BaseConstraintEnv].
* Monitoring wrappers add structured dictionaries under `info["constraint"]`
  and/or `info["metrics"]`.
* Vector wrappers in this file use a simple Python list API:
  observations, rewards, terminals, truncations, infos are lists of length
  `VecEnvWrapperBase.n_envs`.

### Notes
For potential-based shaping, the shaped *cost* inserted into `info` is of the
form

$$
c'_t \;=\; c_t \;+\; \gamma \Phi(q_{t+1}) \;-\; \Phi(q_t),
$$

where $q_t$ is the DFA state, $c_t$ is the original constraint cost,
$\Phi$ is the potential function, and $\gamma$ is the shaping
discount factor.

## API Reference

### Base Class

::: masa.common.wrappers.ConstraintPersistentWrapper
    options:
      members: true
      filters: ["!^__"]

::: masa.common.wrappers.ConstraintPersistentObsWrapper
    options:
      members: true
      filters: ["!^__"]

## Helpers

::: masa.common.wrappers.is_wrapped
    options:
      members: false

::: masa.common.wrappers.get_wrapped
    options:
      members: false

## Next Steps

- [Core Wrappers](Wrappers/Core%20Wrappers.md) - API reference for core wrappers.
- [Misc Wrappers](Wrappers/Misc%20Wrappers.md) - API reference for miscellanious wrappers.
- [Vectorized Envs](Wrappers/Vectorized%20Envs.md) - API refernce for synchronous vectorized environments and wrappers.
