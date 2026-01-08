# Wrappers

```{eval-rst}
Environment wrappers for MASA-Safe-RL.

This module contains small, composable :class:`gymnasium.Wrapper` utilities that
(1) preserve access to constraint-related objects through wrapper chains,
(2) inject monitoring/metrics into ``info``, (3) apply potential-based reward
shaping for DFA-based constraints, and (4) provide basic observation/reward
normalization and light-weight vector-environment helpers.

Key conventions
~~~~~~~~~~~~~~~
* Constraint-enabled environments expose a ``_constraint`` object and (often)
  ``label_fn`` / ``cost_fn`` attributes. See :class:`masa.common.constraints.base.BaseConstraintEnv`.
* Monitoring wrappers add structured dictionaries under ``info["constraint"]``
  and/or ``info["metrics"]``.
* Vector wrappers in this file use a simple Python list API:
  observations, rewards, terminals, truncations, infos are lists of length
  :attr:`VecEnvWrapperBase.n_envs`.

Notes
~~~~~

For potential-based shaping, the shaped *cost* inserted into ``info`` is of the
form

.. math::

   c'_t \;=\; c_t \;+\; \gamma \Phi(q_{t+1}) \;-\; \Phi(q_t),

where :math:`q_t` is the DFA state, :math:`c_t` is the original constraint cost,
:math:`\Phi` is the potential function, and :math:`\gamma` is the shaping
discount factor.
```

## API Reference

### Base Class

```{eval-rst}
.. autoclass:: masa.common.wrappers.ConstraintPersistentWrapper
   :members:
   :special-members: _constraint,
   :private-members:
   :show-inheritance:
.. autoclass:: masa.common.wrappers.ConstraintPersistentObsWrapper
   :members:
   :special-members: _get_obs
   :private-members:
   :show-inheritance:
```

## Helpers

```{eval-rst}
.. automethod:: masa.common.wrappers.is_wrapped
.. automethod:: masa.common.wrappers.get_wrapped
```

## Next Steps

- [Core Wrappers](Wrappers/Core%20Wrappers) - API reference for core wrappers.
- [Misc Wrappers](Wrappers/Misc%20Wrappers) - API reference for miscellanious wrappers.
- [Vectorized Envs](Wrappers/Vectorized%20Envs) - API refernce for synchronous vectorized environments and wrappers.

```{toctree}
:caption: Wrappers
:hidden:

Wrappers/Core Wrappers
Wrappers/Misc Wrappers
Wrappers/Vectorized Envs
```