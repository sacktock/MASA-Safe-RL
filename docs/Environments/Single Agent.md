# Single Agent

Single-agent environments in MASA use the **Gymnasium** API. They can be used directly as Gymnasium environments, or through
[`masa.common.utils.make_env`](../Get%20Started/Basic%20Usage) when you want the standard MASA wrapper stack for labels,
constraints, and monitoring.

The current collection spans three broad settings:

- **Continuous control**: Cartpole with continuous actions.
- **Discrete state-action control**: a discrete-action Cartpole variant and several finite-state benchmark environments.
- **Tabular environments**: gridworlds, Pacman variants, and the Media Streaming MDP.

## Environment Summary

| Environment ID | Family | Observation space | Action space | Reward signal | Default cost signal |
| --- | --- | --- | --- | --- | --- |
| `cont_cartpole` | Continuous control | `Box(4,)` | `Box(1,)` | `1.0` per stable step | `1.0` outside the stable set |
| `disc_cartpole` | Discrete-action control | `Box(4,)` | `Discrete(2)` | `1.0` per stable step | `1.0` outside the stable set |
| `mini_pacman` | Tabular maze | `Discrete(9248)` | `Discrete(5)` | `1.0` when the food is collected | `1.0` on ghost collision |
| `pacman` | Tabular maze | `Discrete(262088)` | `Discrete(5)` | `1.0` when the food is collected | `1.0` on ghost collision |
| `mini_pacman_with_coins` | Structured discrete maze | `Box(7, 10, 9)` | `Discrete(5)` | coin collection reward | `1.0` on ghost collision |
| `pacman_with_coins` | Structured discrete maze | `Box(15, 19, 9)` | `Discrete(5)` | coin collection reward | `1.0` on ghost collision |
| `colour_grid_world` | Tabular gridworld | `Discrete(81)` | `Discrete(5)` | `1.0` on the goal state | `1.0` on blue |
| `colour_bomb_grid_world` | Tabular gridworld | `Discrete(81)` | `Discrete(5)` | `1.0` on a terminal coloured goal | `1.0` on bomb |
| `colour_bomb_grid_world_v2` | Tabular gridworld | `Discrete(225)` | `Discrete(5)` | `1.0` on any coloured goal | `1.0` on bomb |
| `colour_bomb_grid_world_v3` | Tabular gridworld | `Discrete(1125)` | `Discrete(5)` | `1.0` when the active zone matches the reached colour | `1.0` on bomb |
| `bridge_crossing` | Tabular gridworld | `Discrete(400)` | `Discrete(5)` | `1.0` on goal | `1.0` on lava |
| `bridge_crossing_v2` | Tabular gridworld | `Discrete(400)` | `Discrete(5)` | `1.0` on goal | `1.0` on lava |
| `media_streaming` | Tabular queueing MDP | `Discrete(20)` | `Discrete(2)` | `0.0` or `-1.0` depending on bitrate choice | `1.0` when the buffer is empty |

For the environments that expose model structure in addition to the Gymnasium step API, the access pattern differs slightly:

- Full transition matrix: all gridworlds, `media_streaming`, `mini_pacman`, and `mini_pacman_with_coins`.
- Successor-state dictionary: `pacman` and `pacman_with_coins`.


```{toctree}
:hidden:

Single Agent/Cartpole
Single Agent/Pacman
Single Agent/Gridworlds
Single Agent/Media Streaming
```
