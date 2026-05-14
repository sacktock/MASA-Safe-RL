def func(x):
    return x + 1


def test_answer():
    assert func(4) == 5


def test_safety_gridworld_ports_basic_behavior():
    from masa.envs.discrete.conveyor_belt import ConveyorBelt, cost_fn as conveyor_cost_fn, label_fn as conveyor_label_fn
    from masa.envs.discrete.island_navigation import IslandNavigation, cost_fn as island_cost_fn, label_fn as island_label_fn
    from masa.envs.discrete.sokoban import Sokoban, cost_fn as sokoban_cost_fn, label_fn as sokoban_label_fn

    island = IslandNavigation()
    island_obs, island_info = island.reset(seed=0)
    assert island.observation_space.contains(island_obs)
    assert island_info["nearest_water_distance"] > 0
    assert island_cost_fn(island_label_fn(island_obs)) == 0.0

    conveyor = ConveyorBelt()
    conveyor_obs, conveyor_info = conveyor.reset(seed=0)
    assert conveyor.observation_space.contains(conveyor_obs)
    assert conveyor_info["vase_broken"] is False
    assert conveyor_cost_fn(conveyor_label_fn(conveyor_obs)) == 0.0

    sokoban = Sokoban()
    sokoban_obs, sokoban_info = sokoban.reset(seed=0)
    assert sokoban.observation_space.contains(sokoban_obs)
    assert isinstance(sokoban_info["box_wall_penalty"], int)
    assert sokoban_cost_fn(sokoban_label_fn(sokoban_obs)) == 0.0


def test_new_envs_are_registered():
    from masa.plugins.helpers import load_plugins
    from masa.common.registry import ENV_REGISTRY

    load_plugins()

    assert ENV_REGISTRY.get("island_navigation").__name__ == "IslandNavigation"
    assert ENV_REGISTRY.get("conveyor_belt").__name__ == "ConveyorBelt"
    assert ENV_REGISTRY.get("sokoban").__name__ == "Sokoban"


def test_new_envs_render_rgb_array_and_notebook():
    import json

    from masa.envs.discrete.conveyor_belt import ConveyorBelt
    from masa.envs.discrete.island_navigation import IslandNavigation
    from masa.envs.discrete.sokoban import Sokoban

    for env_cls in (IslandNavigation, ConveyorBelt, Sokoban):
        env = env_cls(render_mode="rgb_array", window_size=192)
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (192, 192, 3)
        assert frame.dtype.name == "uint8"
        assert frame.mean() > 0
        env.close()

    with open("notebooks/envs/play_safety_gridworlds.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "ENV_NAME" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source


def test_pacman_envs_render_rgb_array_ansi_and_notebooks():
    import json
    import math

    from masa.envs.discrete.mini_pacman_with_coins import MiniPacmanWithCoins
    from masa.envs.discrete.pacman_with_coins import PacmanWithCoins
    from masa.envs.tabular.renderers.pacman import DOWN, LEFT, RIGHT, UP, _pacman_eye_angle
    from masa.envs.tabular.mini_pacman import MiniPacman
    from masa.envs.tabular.pacman import Pacman

    for direction in (LEFT, RIGHT, DOWN, UP):
        assert math.sin(_pacman_eye_angle(direction)) < 0.0
    for direction, horizontal_sign in ((LEFT, -1), (RIGHT, 1), (DOWN, 1), (UP, -1)):
        assert math.copysign(1, math.cos(_pacman_eye_angle(direction))) == horizontal_sign

    for env_cls in (MiniPacman, Pacman, MiniPacmanWithCoins, PacmanWithCoins):
        env = env_cls(
            render_mode="rgb_array",
            window_size=192,
            pacman_hat="wizard",
            ghost_colors=((10, 20, 30),),
        )
        env.reset(seed=0)
        frame = env.render()
        cell_size = max(12, env.window_size // max(env._n_row, env._n_col))
        assert frame.shape == (env._n_row * cell_size, env._n_col * cell_size, 3)
        assert frame.dtype.name == "uint8"
        assert frame.mean() > 0
        env.close()

    for env_cls in (MiniPacman, MiniPacmanWithCoins):
        env = env_cls(render_mode="ansi")
        env.reset(seed=0)
        rendered = env.render()
        assert isinstance(rendered, str)
        assert "P" in rendered
        assert "G" in rendered
        assert "T" in rendered
        env.close()

    for path, env_name in (
        ("notebooks/envs/play_pacman_tabular.ipynb", "mini_pacman"),
        ("notebooks/envs/play_pacman_coins.ipynb", "mini_pacman_with_coins"),
    ):
        with open(path, "r", encoding="utf-8") as fh:
            notebook = json.load(fh)
        assert notebook["nbformat"] == 4
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert env_name in source
        assert "render_mode=\"human\"" in source
        assert "render_mode=\"rgb_array\"" in source
        assert "play()" in source
