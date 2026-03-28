# Cartpole

MASA provides two single-agent Cartpole environments:

- `disc_cartpole`: discrete actions.
- `cont_cartpole`: continuous actions.

Both variants implement the same dynamics and state representation. They differ only in how the control input is specified.

## State, Labels, and Costs

The observation is a 4-dimensional vector

```text
[x, x_dot, theta, theta_dot]
```

where `x` is the cart position and `theta` is the pole angle. Both environments use the same default labelling and cost functions:

- `label_fn(obs)` returns `{"stable"}` exactly when `|theta| <= 0.2095` and `|x| <= 2.4`.
- `cost_fn(labels)` returns `0.0` when `"stable"` is present and `1.0` otherwise.

This makes Cartpole a useful benchmark for state-based safety specifications: the reward encourages long survival, while the cost
marks transitions that leave the stable region.

## Dynamics

The implementation follows the standard Cartpole equations of motion with:

- gravity `9.8`
- pole length `0.5`
- force magnitude `10.0`
- time step `0.02`
- Euler integration

At reset, the state is sampled uniformly from `[-0.05, 0.05]^4`.

Each step returns reward `1.0`. The environment terminates as soon as the state leaves the stable set described above. It does not
internally truncate episodes, so practical runs usually apply `TimeLimit` on top.

## Variant Differences

### `disc_cartpole`

- Action space: `Discrete(2)`.
- Action `0` applies force `-10.0`.
- Action `1` applies force `+10.0`.

This is the most direct fit for discrete-action safe RL methods.

### `cont_cartpole`

- Action space: `Box(low=-1.0, high=1.0, shape=(1,))`.
- The scalar action is scaled by the force magnitude, so the applied force is `10.0 * action[0]`.

This variant is useful for continuous-control algorithms such as PPO or A2C with continuous policies.

## Practical Use

Both environments are registered in MASA and can be created through `make_env`:

```python
from masa.common.utils import make_env
from masa.envs.discrete.cartpole import label_fn, cost_fn

env = make_env(
    "disc_cartpole",
    "cmdp",
    500,
    label_fn=label_fn,
    cost_fn=cost_fn,
    budget=0.0,
)
```

For the continuous variant, replace the environment id and import path with `cont_cartpole` and
`masa.envs.continuous.cartpole`.
