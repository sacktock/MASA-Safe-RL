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
