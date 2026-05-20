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
    import masa
    from masa.common.registry import ENV_REGISTRY

    assert ENV_REGISTRY.get("IslandNavigation").__name__ == "IslandNavigation"
    assert ENV_REGISTRY.get("ConveyorBelt").__name__ == "ConveyorBelt"
    assert ENV_REGISTRY.get("Sokoban").__name__ == "Sokoban"


def test_new_envs_render_rgb_array_and_notebook():
    import json

    from masa.envs.discrete.conveyor_belt import ConveyorBelt
    from masa.envs.discrete.island_navigation import IslandNavigation
    from masa.envs.discrete.sokoban import Sokoban

    for env_cls in (IslandNavigation, ConveyorBelt, Sokoban):
        env = env_cls(render_mode="rgb_array", render_window_size=192)
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
    assert "widgets.ToggleButtons" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=512" in source


def test_pacman_envs_render_rgb_array_ansi_and_notebooks(monkeypatch):
    import sys
    import json
    import math
    from collections import defaultdict

    import numpy as np

    from masa.envs.tabular import utils as pacman_utils
    from masa.envs.tabular.renderers.pacman import DOWN, LEFT, RIGHT, UP, _pacman_eye_angle

    def fake_pacman_transition_dict(
        standard_map,
        return_matrix=False,
        n_directions=4,
        n_actions=5,
        n_ghosts=1,
        ghost_rand_prob=0.6,
        food_x=None,
        food_y=None,
    ):
        del n_directions, n_ghosts, ghost_rand_prob
        if standard_map.shape == (7, 10):
            state = (1, 4, 1, 5, 3, 1, 1 if food_x is not None and food_y is not None else 0)
        else:
            state = (7, 1, 1, 7, 12, 3, 1 if food_x is not None and food_y is not None else 0)
        state_map = {state: 0}
        reverse_state_map = {0: state}
        successor_states = defaultdict(list, {0: [0]})
        transition_probs = defaultdict(lambda: np.array([1.0], dtype=np.float32))
        matrix = None
        if return_matrix:
            matrix = np.ones((1, 1, n_actions), dtype=np.float32)
        return successor_states, transition_probs, matrix, 1, state_map, reverse_state_map

    monkeypatch.setattr(pacman_utils, "create_pacman_transition_dict", fake_pacman_transition_dict)
    for module in (
        "masa.envs.tabular.mini_pacman",
        "masa.envs.tabular.pacman",
        "masa.envs.discrete.mini_pacman_with_coins",
        "masa.envs.discrete.pacman_with_coins",
    ):
        sys.modules.pop(module, None)

    from masa.envs.discrete.mini_pacman_with_coins import MiniPacmanWithCoins
    from masa.envs.discrete.pacman_with_coins import PacmanWithCoins
    from masa.envs.tabular.mini_pacman import MiniPacman
    from masa.envs.tabular.pacman import Pacman

    for direction in (LEFT, RIGHT, DOWN, UP):
        assert math.sin(_pacman_eye_angle(direction)) < 0.0
    for direction, horizontal_sign in ((LEFT, -1), (RIGHT, 1), (DOWN, 1), (UP, -1)):
        assert math.copysign(1, math.cos(_pacman_eye_angle(direction))) == horizontal_sign

    for env_cls in (MiniPacman, Pacman, MiniPacmanWithCoins, PacmanWithCoins):
        env = env_cls(
            render_mode="rgb_array",
            render_window_size=192,
            pacman_hat="wizard",
            ghost_colors=((10, 20, 30),),
        )
        env.reset(seed=0)
        frame = env.render()
        cell_size = max(12, env.render_window_size // max(env._n_row, env._n_col))
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
        assert "widgets.ToggleButtons" in source
        assert "render_mode=\"human\"" in source
        assert "render_mode=\"rgb_array\"" in source
        assert "render_window_size=512" in source
        assert "def make_env" in source
        assert "play()" in source


def test_cartpole_envs_render_rgb_array_ansi_and_notebook():
    import json

    import numpy as np
    import pytest

    from masa.envs.continuous.cartpole import ContinuousCartpole
    from masa.envs.discrete.cartpole import DiscreteCartpole

    env_cases = (
        (DiscreteCartpole, 1),
        (ContinuousCartpole, np.array([1.0], dtype=np.float32)),
    )

    for env_cls, action in env_cases:
        assert env_cls.metadata["render_fps"] == 30
        env = env_cls(render_mode="rgb_array", render_window_size=192)
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (192, 192, 3)
        assert frame.dtype.name == "uint8"
        assert frame.mean() > 0
        env.step(action)
        next_frame = env.render()
        assert next_frame.shape == frame.shape
        assert next_frame.mean() > 0
        env.close()

    for env_cls, action in env_cases:
        env = env_cls(render_mode="ansi")
        env.reset(seed=0)
        env.step(action)
        rendered = env.render()
        assert isinstance(rendered, str)
        for token in ("cart", "pole", "status", "last_action"):
            assert token in rendered
        env.close()

    for env_cls, _ in env_cases:
        env = env_cls()
        env.reset(seed=0)
        assert env.render() is None
        env.close()

        with pytest.raises(ValueError):
            env_cls(render_mode="bad")
        with pytest.raises(ValueError):
            env_cls(render_window_size=0)

    with open("notebooks/envs/play_cartpole.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "DiscreteCartpole" in source
    assert "ContinuousCartpole" in source
    assert "widgets.ToggleButtons" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=512" in source
    assert "pygame.K_LEFT" in source
    assert "pygame.K_SPACE" in source
    assert "np.array([0.0]" in source
    assert "pygame.key.get_pressed()" in source
    assert "play_env" in source


def test_mountain_car_envs_render_rgb_array_ansi():
    import json

    import numpy as np
    import pytest

    from masa.envs.continuous.mountain_car import ContinuousMountainCar
    from masa.envs.continuous.renderers import mountain_car as mountain_car_renderer
    from masa.envs.discrete.base import DiscreteEnv
    from masa.envs.discrete.mountain_car import DiscreteMountainCar

    assert issubclass(DiscreteMountainCar, DiscreteEnv)

    env_cases = (
        (DiscreteMountainCar, 1),
        (ContinuousMountainCar, np.array([1.0], dtype=np.float32)),
    )

    for env_cls, action in env_cases:
        assert env_cls.metadata["render_fps"] == 60
        env = env_cls(render_mode="rgb_array", render_window_size=192)
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (192, 192, 3)
        assert frame.dtype.name == "uint8"
        assert frame.mean() > 0
        np.testing.assert_allclose(frame[0, 0], mountain_car_renderer.PANEL_COLOR, atol=4)
        np.testing.assert_allclose(frame[0, -1], mountain_car_renderer.PANEL_COLOR, atol=4)
        env.step(action)
        next_frame = env.render()
        assert next_frame.shape == frame.shape
        assert next_frame.mean() > 0
        env.close()

    for env_cls, action in env_cases:
        env = env_cls(render_mode="ansi")
        env.reset(seed=0)
        env.step(action)
        rendered = env.render()
        assert isinstance(rendered, str)
        for token in ("position", "velocity", "status", "last_action"):
            assert token in rendered
        env.close()

    for env_cls, _ in env_cases:
        env = env_cls()
        env.reset(seed=0)
        assert env.render() is None
        env.close()

        with pytest.raises(ValueError):
            env_cls(render_mode="bad")
        with pytest.raises(ValueError):
            env_cls(render_window_size=0)

    env = DiscreteMountainCar(render_mode="rgb_array", render_window_size=192)
    env.reset(seed=0)
    snapshot = env._renderer._snapshot()
    scale = mountain_car_renderer._gym_scale(snapshot, env.render_window_size)
    body_points, _, wheel_centers, _ = mountain_car_renderer._car_geometry(snapshot, scale, env.render_window_size)
    assert wheel_centers[0][1] > sum(point[1] for point in body_points) / len(body_points)
    assert wheel_centers[1][1] > sum(point[1] for point in body_points) / len(body_points)
    hill_points = mountain_car_renderer._hill_points(snapshot, scale, env.render_window_size)
    assert hill_points[0][0] == 0
    assert hill_points[-1][0] == env.render_window_size
    env.close()

    with open("notebooks/envs/play_mountain_car.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "DiscreteMountainCar" in source
    assert "ContinuousMountainCar" in source
    assert "widgets.ToggleButtons" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=512" in source
    assert "pygame.K_LEFT" in source
    assert "pygame.K_SPACE" in source
    assert "pygame.key.get_pressed()" in source
    assert "play_env" in source


def test_road_2d_render_rgb_array_ansi():
    import numpy as np
    import pytest

    from masa.envs.continuous.road_2d import Road2D

    action = np.array([1.0, 1.0], dtype=np.float32)

    env = Road2D(render_mode="rgb_array", render_window_size=192)
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (192, 192, 3)
    assert frame.dtype.name == "uint8"
    assert frame.mean() > 0
    env.step(action)
    next_frame = env.render()
    assert next_frame.shape == frame.shape
    assert next_frame.mean() > 0
    env.close()

    env = Road2D(render_mode="ansi")
    env.reset(seed=0)
    env.step(action)
    rendered = env.render()
    assert isinstance(rendered, str)
    for token in ("position", "velocity", "status", "last_action"):
        assert token in rendered
    env.close()

    env = Road2D()
    env.reset(seed=0)
    assert env.render() is None
    env.close()

    with pytest.raises(ValueError):
        Road2D(render_mode="bad")
    with pytest.raises(ValueError):
        Road2D(render_window_size=0)


def test_road_1d_render_rgb_array_ansi():
    import json

    import numpy as np
    import pytest

    from masa.envs.continuous.road_1d import Road1D

    action = np.array([1.0], dtype=np.float32)

    env = Road1D(render_mode="rgb_array", render_window_size=192)
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (192, 192, 3)
    assert frame.dtype.name == "uint8"
    assert frame.mean() > 0
    env.step(action)
    next_frame = env.render()
    assert next_frame.shape == frame.shape
    assert next_frame.mean() > 0
    env.close()

    env = Road1D(render_mode="ansi")
    env.reset(seed=0)
    env.step(action)
    rendered = env.render()
    assert isinstance(rendered, str)
    for token in ("position", "velocity", "status", "last_action"):
        assert token in rendered
    env.close()

    env = Road1D()
    env.reset(seed=0)
    assert env.render() is None
    env.close()

    with pytest.raises(ValueError):
        Road1D(render_mode="bad")
    with pytest.raises(ValueError):
        Road1D(render_window_size=0)

    with open("notebooks/envs/play_roads.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Road1D" in source
    assert "Road2D" in source
    assert "widgets.ToggleButtons" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=512" in source
    assert "pygame.K_LEFT" in source
    assert "pygame.K_SPACE" in source
    assert "play_env" in source


def test_obstacle_envs_render_rgb_array_ansi():
    import json

    import numpy as np
    import pytest

    from masa.envs.continuous.obstacle import Obstacle
    from masa.envs.continuous.obstacle_v2 import ObstacleV2
    from masa.envs.continuous.obstacle_v3 import ObstacleV3
    from masa.envs.continuous.obstacle_v4 import ObstacleV4

    action = np.array([1.0, 1.0], dtype=np.float32)
    env_classes = (Obstacle, ObstacleV2, ObstacleV3, ObstacleV4)

    for env_cls in env_classes:
        env = env_cls(render_mode="rgb_array", render_window_size=192)
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (192, 192, 3)
        assert frame.dtype.name == "uint8"
        assert frame.mean() > 0
        env.step(action)
        next_frame = env.render()
        assert next_frame.shape == frame.shape
        assert next_frame.mean() > 0
        env.close()

    for env_cls in env_classes:
        env = env_cls(render_mode="ansi")
        env.reset(seed=0)
        env.step(action)
        rendered = env.render()
        assert isinstance(rendered, str)
        for token in ("position", "velocity", "status", "last_action"):
            assert token in rendered
        env.close()

    for env_cls in env_classes:
        env = env_cls()
        env.reset(seed=0)
        assert env.render() is None
        env.close()

        with pytest.raises(ValueError):
            env_cls(render_mode="bad")
        with pytest.raises(ValueError):
            env_cls(render_window_size=0)

    with open("notebooks/envs/play_obstacles.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for env_name in ("Obstacle", "ObstacleV2", "ObstacleV3", "ObstacleV4"):
        assert env_name in source
    assert "widgets.ToggleButtons" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=512" in source
    assert "pygame.K_LEFT" in source
    assert "pygame.K_SPACE" in source
    assert "play_env" in source


def test_colour_grid_world_render_rgb_array_ansi_and_notebook():
    import json

    import pytest

    from masa.envs.tabular.colour_grid_world import ColourGridWorld

    env = ColourGridWorld(render_mode="rgb_array", render_window_size=192)
    env.reset(seed=0)
    frame = env.render()
    cell_size = max(12, env.render_window_size // env._grid_size)
    assert frame.shape == (env._grid_size * cell_size, env._grid_size * cell_size, 3)
    assert frame.dtype.name == "uint8"
    assert frame.mean() > 0
    env.close()

    env = ColourGridWorld(render_mode="ansi")
    env.reset(seed=0)
    env._state = 1
    rendered = env.render()
    assert isinstance(rendered, str)
    for marker in ("A", "S", "T", "X", "G", "P"):
        assert marker in rendered
    env.close()

    env = ColourGridWorld()
    env.reset(seed=0)
    assert env.render() is None
    env.close()

    with pytest.raises(ValueError):
        ColourGridWorld(render_mode="bad")
    with pytest.raises(ValueError):
        ColourGridWorld(render_window_size=0)

    with open("notebooks/envs/play_colour_grid_world.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "ENV_NAME" in source
    assert "colour_grid_world" in source
    assert "ColourGridWorld" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=512" in source
    assert "pygame.K_SPACE: 4" in source
    assert "play_env" in source


def test_colour_bomb_envs_render_rgb_array_ansi_and_notebook():
    import json

    import pytest

    from masa.envs.tabular.colour_bomb_grid_world import ColourBombGridWorld
    from masa.envs.tabular.colour_bomb_grid_world_v2 import ColourBombGridWorldV2
    from masa.envs.tabular.colour_bomb_grid_world_v3 import ColourBombGridWorldV3

    env_classes = (ColourBombGridWorld, ColourBombGridWorldV2, ColourBombGridWorldV3)

    for env_cls in env_classes:
        env = env_cls(render_mode="rgb_array", render_window_size=192)
        env.reset(seed=0)
        frame = env.render()
        cell_size = max(12, env.render_window_size // env._grid_size)
        assert frame.shape == (env._grid_size * cell_size, env._grid_size * cell_size, 3)
        assert frame.dtype.name == "uint8"
        assert frame.mean() > 0
        env.close()

    for env_cls in env_classes:
        env = env_cls(render_mode="ansi")
        env.reset(seed=0)
        rendered = env.render()
        assert isinstance(rendered, str)
        assert "A" in rendered
        assert "#" in rendered
        assert "X" in rendered
        assert any(marker in rendered for marker in ("G", "Y", "R", "B", "P"))
        env.close()

    for env_cls in (ColourBombGridWorldV2, ColourBombGridWorldV3):
        env = env_cls(render_mode="ansi")
        env.reset(seed=0)
        rendered = env.render()
        assert "M" in rendered
        env.close()

    for env_cls in env_classes:
        env = env_cls()
        env.reset(seed=0)
        assert env.render() is None
        env.close()

        with pytest.raises(ValueError):
            env_cls(render_mode="bad")
        with pytest.raises(ValueError):
            env_cls(render_window_size=0)

    with open("notebooks/envs/play_colour_bomb_gridworlds.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "colour_bomb_grid_world" in source
    assert "colour_bomb_grid_world_v2" in source
    assert "colour_bomb_grid_world_v3" in source
    assert "widgets.ToggleButtons" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=512" in source
    assert "pygame.K_SPACE: 4" in source
    assert "def make_env" in source
    assert "play_env" in source


def test_colour_bomb_wall_states_are_impassable_under_slip():
    import numpy as np

    from masa.envs.tabular.colour_bomb_grid_world import ColourBombGridWorld
    from masa.envs.tabular.colour_bomb_grid_world_v2 import ColourBombGridWorldV2
    from masa.envs.tabular.colour_bomb_grid_world_v3 import ColourBombGridWorldV3

    for env_cls in (ColourBombGridWorld, ColourBombGridWorldV2, ColourBombGridWorldV3):
        env = env_cls()
        wall_states = sorted(getattr(env, "_wall_states", []))
        non_wall_states = [state for state in range(env._n_states) if state not in wall_states]

        wall_entry_probs = env._transition_matrix[wall_states][:, non_wall_states, :]
        assert np.allclose(wall_entry_probs, 0.0)


def test_media_streaming_render_rgb_array_ansi_and_notebook():
    import json

    import pytest

    from masa.envs.tabular.media_streaming import MediaStreaming

    env = MediaStreaming(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (320, 640, 3)
    assert frame.dtype.name == "uint8"
    assert frame.mean() > 0
    env.close()

    env = MediaStreaming(render_mode="rgb_array", render_window_size=192)
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (160, 192, 3)
    assert frame.dtype.name == "uint8"
    assert frame.mean() > 0
    env.step(1)
    next_frame = env.render()
    assert next_frame.shape == frame.shape
    assert next_frame.mean() > 0
    env.close()

    env = MediaStreaming(render_mode="ansi")
    env.reset(seed=0)
    rendered = env.render()
    assert isinstance(rendered, str)
    for marker in ("A", "S", "E"):
        assert marker in rendered
    env.close()

    env = MediaStreaming()
    env.reset(seed=0)
    assert env.render() is None
    env.close()

    with pytest.raises(ValueError):
        MediaStreaming(render_mode="bad")
    with pytest.raises(ValueError):
        MediaStreaming(render_window_size=0)

    with open("notebooks/envs/play_media_streaming.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "ENV_NAME" in source
    assert "media_streaming" in source
    assert "MediaStreaming" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=640" in source
    assert "pygame.K_LEFT: 0" in source
    assert "pygame.K_SPACE: 1" in source
    assert "play_env" in source


def test_bridge_crossing_envs_render_rgb_array_ansi_and_notebook():
    import json

    import pytest

    from masa.envs.tabular.bridge_crossing import BridgeCrossing
    from masa.envs.tabular.bridge_crossing_v2 import BridgeCrossingV2

    env_classes = (BridgeCrossing, BridgeCrossingV2)

    for env_cls in env_classes:
        env = env_cls(render_mode="rgb_array", render_window_size=192)
        env.reset(seed=0)
        frame = env.render()
        cell_size = max(12, env.render_window_size // env._grid_size)
        assert frame.shape == (env._grid_size * cell_size, env._grid_size * cell_size, 3)
        assert frame.dtype.name == "uint8"
        assert frame.mean() > 0
        env.close()

    for env_cls in env_classes:
        env = env_cls(render_mode="ansi")
        env.reset(seed=0)
        env._state = env._start_state + 1
        rendered = env.render()
        assert isinstance(rendered, str)
        for marker in ("A", "S", "G", "L"):
            assert marker in rendered
        env.close()

    for env_cls in env_classes:
        env = env_cls()
        env.reset(seed=0)
        assert env.render() is None
        env.close()

        with pytest.raises(ValueError):
            env_cls(render_mode="bad")
        with pytest.raises(ValueError):
            env_cls(render_window_size=0)

    with open("notebooks/envs/play_bridge_crossing.ipynb", "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "bridge_crossing" in source
    assert "bridge_crossing_v2" in source
    assert "widgets.ToggleButtons" in source
    assert "render_mode=\"human\"" in source
    assert "render_mode=\"rgb_array\"" in source
    assert "render_window_size=512" in source
    assert "pygame.K_SPACE: 4" in source
    assert "def make_env" in source
    assert "play_env" in source
