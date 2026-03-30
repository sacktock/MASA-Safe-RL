# Gridworlds

MASA includes several single-agent tabular gridworlds. They all use discrete states, discrete actions, and explicit stochastic
transition models. Every environment in this family exposes a full transition matrix via `get_transition_matrix()`.

## Shared Action Convention

The gridworld helpers use the same five actions throughout:

- `0`: move left
- `1`: move right
- `2`: move down
- `3`: move up
- `4`: stay in place

When slip is enabled, the intended action is taken with high probability and the remaining probability mass is spread uniformly over
the other actions.

## Available Environments

- [Bridge Crossing](Gridworlds/Bridge%20Crossing)
- [Colour Grid World](Gridworlds/Colour%20Grid%20World)
- [Colour Bomb Grid World](Gridworlds/Colour%20Bomb%20Grid%20World)

```{toctree}
:hidden:

Gridworlds/Bridge Crossing
Gridworlds/Colour Grid World
Gridworlds/Colour Bomb Grid World
```
