# Vectorization and Normalization

This tutorial demonstrates MASA's observation, vectorization, and normalization wrappers without training a policy.

Runnable notebook: [notebooks/tutorials/11_vectorization_and_normalization.ipynb](../../../notebooks/tutorials/11_vectorization_and_normalization.ipynb)

## Wrapper Map

| Wrapper | Input | Output | Use it when |
| --- | --- | --- | --- |
| `OneHotObsWrapper` | `Discrete` observations, or `Dict` observations containing `Discrete` leaves | `Box` one-hot vectors | an algorithm expects vector observations |
| `FlattenDictObsWrapper` | `Dict` observations whose values are already `Box` spaces | one flat `Box` | a wrapped environment emits structured observation pieces |
| `DummyVecWrapper` | one environment | vector API with length-1 lists | code expects vectorized reset and step outputs |
| `VecWrapper` | a list of environments | synchronous batched list outputs | you want to step several environments together |
| `NormWrapper` | one `Box`-observation environment | normalized observations and/or rewards | you want running statistics for a single environment |
| `VecNormWrapper` | a `DummyVecWrapper` or `VecWrapper` with `Box` observations | vectorized normalized observations and/or rewards | you want running statistics across parallel environments |

The notebook focuses on stable `reset` and `step` behavior. It does not cover `reset_done`.

## One-Hot Discrete Observations

`colour_grid_world` emits a discrete state id:

```python
env = make_env(
    "colour_grid_world",
    "cmdp",
    5,
    label_fn=colour_grid_label_fn,
    cost_fn=colour_grid_cost_fn,
    budget=0.0,
)
```

The raw observation space is `Discrete(81)`. After `OneHotObsWrapper`, the observation is a `Box(81,)` vector with exactly one active entry.

This is the usual first step before feeding a tabular environment into wrappers or algorithms that expect vector observations.

## Vector APIs

`DummyVecWrapper` gives one environment the vectorized interface:

```python
vec_env = DummyVecWrapper(OneHotObsWrapper(make_colour_grid_env()))
obs, info = vec_env.reset(seed=0)
```

The reset and step results are lists of length `1`.

`VecWrapper` steps multiple environments synchronously:

```python
vec_env = VecWrapper(
    [OneHotObsWrapper(make_colour_grid_env()) for _ in range(2)]
)
obs, infos = vec_env.reset(seed=10)
obs, rewards, terminated, truncated, infos = vec_env.step([0, 1])
```

The result lists have one entry per environment.

## Normalization

`NormWrapper` is for a single `Box`-observation environment. The tutorial uses `cont_cartpole` with `pctl`:

```python
env = NormWrapper(
    make_cartpole_env(),
    norm_obs=True,
    norm_rew=False,
    training=True,
)
```

With `norm_obs=True`, the wrapper updates running observation statistics and returns normalized observations. With `norm_rew=False`, rewards stay in their original scale.

`VecNormWrapper` applies the same idea to a vectorized environment:

```python
env = VecNormWrapper(
    VecWrapper([make_cartpole_env(), make_cartpole_env()]),
    norm_obs=True,
    norm_rew=False,
    training=True,
)
```

Reset returns a batched normalized observation array with shape `(2, 4)`, while rewards and infos still have one entry per environment.

## Flattening Dict Observations

`ltl_safety` environments can expose structured observations. For example, `colour_bomb_grid_world` with `obs_type="dict"` returns:

```text
orig:       the original grid state
automaton: the DFA state
```

Those values are discrete, so the tutorial first applies `OneHotObsWrapper`. That turns the dict leaves into `Box` vectors:

```text
orig       -> Box(81,)
automaton  -> Box(2,)
```

Then `FlattenDictObsWrapper` concatenates them into one flat `Box(83,)` observation.

The ordering matters: `FlattenDictObsWrapper` expects `Box` values at runtime, so discrete dict leaves should be one-hot encoded first.
