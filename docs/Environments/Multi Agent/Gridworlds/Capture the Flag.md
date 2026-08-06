# Capture the Flag

`CaptureTheFlag` is a native two-team PettingZoo `ParallelEnv`. Red and blue agents paint territory, disable opponents with paint beams, steal the opposing flag, and return it to their own flag while that flag is at home. 

Note that this environment was adapted from [MeltingPot v2](https://arxiv.org/pdf/2211.13746).

```python
from masa.envs.multiagent.tabular import CaptureTheFlag

env = CaptureTheFlag(render_mode="rgb_array")
observations, infos = env.reset(seed=0)
```

The default game has eight agents, split evenly between the teams, and a 1,000-step time limit. Both values are configurable. Observations are flat
26-element feature vectors by default; set `flatten_observations=False` for shape `(1, 1, 26)`.

## Actions

- `0`: no-op
- `1`: move forward
- `2`: strafe right
- `3`: strafe left
- `4`: move backward
- `5`: turn left
- `6`: turn right
- `7`: fire the short, wide primary paint beam
- `8`: fire the long, narrow secondary paint beam

The secondary beam only fires after the agent has remained in place for a step. Paint applied by beams affects movement and health regeneration: enemy paint stops movement, friendly paint permits up to three health, and enemy paint limits health to one.

## Observations and rewards

Each observation contains the agent's position, orientation, team, active state, health, respawn timer, weapon readiness, carried flag, both flags' home states, and one-step event indicators for hits, zaps, flag interactions, respawning, and firing. `FEATURE_NAMES` in the environment module gives the exact channel order.

A successful capture gives every teammate `+1` and every opponent `-1`. Other events have zero reward by default. Episodes truncate at `max_episode_steps`.

## Labels and cost

The default `label_fn` describes team, activity, health, flag state, and the one-step events represented in the observation. A hit or a zap received adds the `unsafe` label. The default `cost_fn` is therefore:

```python
cost = 1.0 if "unsafe" in labels else 0.0
```

The cost is event-based: an inactive agent remains labelled `respawning`, but does not incur another cost on every respawn frame.

Rendering supports `ansi`, `rgb_array`, and `human`. A playable example is in `notebooks/envs/multiagent/play_capture_the_flag.ipynb`.
