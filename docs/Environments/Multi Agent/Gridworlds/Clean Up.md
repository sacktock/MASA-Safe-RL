# Clean Up

`CleanUp` is a native PettingZoo `ParallelEnv` for the sequential social dilemma in which agents choose between gathering apples and maintaining a
shared river. Dirt suppresses apple growth, while cleaning benefits every agent and does not directly reward the cleaner.

Note that this environment was adapted from [MeltingPot v2](https://arxiv.org/pdf/2211.13746).

```python
from masa.envs.multiagent.tabular import CleanUp

env = CleanUp(render_mode="rgb_array")
observations, infos = env.reset(seed=0)
```

The default game has seven agents and a 5,000-step time limit. Observations are flat 18-element feature vectors by default; set `flatten_observations=False` for shape `(1, 1, 18)`.

## Actions

- `0`: no-op
- `1`: move forward
- `2`: strafe right
- `3`: strafe left
- `4`: move backward
- `5`: turn left
- `6`: turn right
- `7`: fire the zap beam
- `8`: fire the cleaning beam

Agents collect a reward of `+1` for each apple consumed. Apple growth falls as the fraction of dirty river cells approaches 40 percent. Dirt begins spawning after 50 steps by default. Zapped agents leave the map temporarily and then respawn.

## Observations

Each observation contains the agent's position, orientation, active state, respawn timer, global dirty and clean river-cell counts, zap readiness, the number of other cleaners, and one-step indicators for cleaning, apple consumption, dirt spawning, firing, being zapped, and respawning.
`FEATURE_NAMES` in the environment module gives the exact channel order.

## Labels and cost

The default `label_fn` exposes activity, cleaning, apple consumption, firing, respawning, and `dirty_world` when at least 40 percent of river cells are dirty. Receiving a zap adds `got_zapped` and `unsafe`. The default `cost_fn` is:

```python
cost = 1.0 if "unsafe" in labels else 0.0
```

As in Capture the Flag, this cost is charged on the event step rather than on every subsequent respawn frame.

Rendering supports `ansi`, `rgb_array`, and `human`. A playable example is in `notebooks/envs/multiagent/play_clean_up.ipynb`.
