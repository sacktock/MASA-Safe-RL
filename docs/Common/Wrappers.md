# Wrappers

## Base Classes

```{eval-rst}
.. automodule:: masa.common.wrappers
   :members: ConstraintPersistentWrapper, ConstraintPersistentObsWrapper
   :special-members: _constraint, _get_obs
   :private-members:
   :show-inheritance:
```

## Core Wrappers

```{eval-rst}
.. autoclass:: masa.common.wrappers.TimeLimit
    :members:
    :show-inheritance:
.. autoclass:: masa.common.wrappers.ConstraintMonitor
    :members:
    :show-inheritance:
    :special-members: _step_metrics, _epsiode_metrics
    :private-members:
.. autoclass:: masa.common.wrappers.RewardMonitor
    :members:
    :show-inheritance:
    :special-members: _epsiode_metrics
    :private-members:
```

## Other Wrappers

```{eval-rst}
.. autoclass:: masa.common.wrappers.RewardShapingWrapper
    :members:
    :show-inheritance:
.. autoclass:: masa.common.wrappers.NormWrapper
    :members:
    :show-inheritance:
.. autoclass:: masa.common.wrappers.OneHotObsWrapper
    :members:
    :show-inheritance:
.. autoclass:: masa.common.wrappers.FlattenDictObsWrapper
    :members:
    :show-inheritance:
```

## Vectorized Envs

```{eval-rst}
.. autoclass:: masa.common.wrappers.VecEnvWrapperBase
    :members:
    :show-inheritance:
.. autoclass:: masa.common.wrappers.DummyVecWrapper
    :members:
    :show-inheritance:
.. autoclass:: masa.common.wrappers.VecWrapper
    :members:
    :show-inheritance:
.. autoclass:: masa.common.wrappers.VecNormWrapper
    :members:
    :show-inheritance:
```

## Helpers

```{eval-rst}
.. automethod:: masa.common.wrappers.is_wrapped
.. automethod:: masa.common.wrappers.get_wrapped
```