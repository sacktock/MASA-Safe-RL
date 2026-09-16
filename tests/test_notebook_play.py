from __future__ import annotations

import json
import threading
from pathlib import Path

from masa.common.rendering.notebook_play import (
    notebook_video_recording,
    start_play_thread,
    stop_play_thread,
    sync_selected_env,
)


class _Selector:
    def __init__(self, value: str) -> None:
        self.value = value


class _Env:
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs
        self.closed = False

    def reset(self, *, seed=None):
        self.seed = seed
        return f"obs:{self.name}", {"name": self.name, "seed": seed}

    def close(self) -> None:
        self.closed = True


class _FakeDisplay:
    def __init__(self) -> None:
        self.quit_called = False

    def get_init(self) -> bool:
        return True

    def quit(self) -> None:
        self.quit_called = True


class _FakeEvent:
    def __init__(self) -> None:
        self.clear_called = False

    def clear(self) -> None:
        self.clear_called = True


class _FakePygame:
    def __init__(self) -> None:
        self.display = _FakeDisplay()
        self.event = _FakeEvent()


def _load_notebook(notebook_path: str) -> dict:
    with Path(notebook_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _notebook_source(notebook_path: str) -> str:
    notebook = _load_notebook(notebook_path)
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_sync_selected_env_reuses_env_when_selection_is_unchanged():
    env = _Env("Obstacle")
    selector = _Selector("Obstacle")

    returned_env, env_name, obs, info, switched = sync_selected_env(
        env,
        "Obstacle",
        selector,
        lambda env_name, **kwargs: _Env(env_name, **kwargs),
        seed=0,
        render_window_size=512,
    )

    assert returned_env is env
    assert env_name == "Obstacle"
    assert obs is None
    assert info is None
    assert switched is False
    assert env.closed is False


def test_sync_selected_env_closes_old_env_and_opens_selected_env():
    old_env = _Env("Obstacle")
    selector = _Selector("ObstacleV3")
    pygame = _FakePygame()
    calls = []

    def make_env(env_name, **kwargs):
        calls.append((env_name, kwargs))
        return _Env(env_name, **kwargs)

    new_env, env_name, obs, info, switched = sync_selected_env(
        old_env,
        "Obstacle",
        selector,
        make_env,
        seed=7,
        render_window_size=384,
        env_kwargs={"difficulty": "demo"},
        pygame=pygame,
    )

    assert switched is True
    assert old_env.closed is True
    assert pygame.event.clear_called is True
    assert pygame.display.quit_called is True
    assert env_name == "ObstacleV3"
    assert new_env.name == "ObstacleV3"
    assert obs == "obs:ObstacleV3"
    assert info == {"name": "ObstacleV3", "seed": 7}
    assert calls == [
        (
            "ObstacleV3",
            {
                "render_mode": "human",
                "render_window_size": 384,
                "difficulty": "demo",
            },
        )
    ]


def test_start_play_thread_stops_previous_session():
    session_key = "test-start-play-thread-stops-previous"
    started = threading.Event()

    def run_until_stopped(stop_event):
        started.set()
        stop_event.wait()

    first = start_play_thread(session_key, run_until_stopped)
    assert started.wait(timeout=1.0)

    started.clear()
    second = start_play_thread(session_key, run_until_stopped)
    try:
        assert first.stop_event.is_set()
        assert not first.is_alive
        assert started.wait(timeout=1.0)
        assert second.is_alive
    finally:
        stop_play_thread(session_key)


def test_disabled_notebook_video_recording_is_a_noop(tmp_path):
    video_path = tmp_path / "disabled.mp4"
    with notebook_video_recording(False, video_path) as recorder:
        assert recorder.enabled is False
    assert not video_path.exists()


def test_selector_notebooks_sync_selected_envs_during_play():
    notebook_paths = (
        "notebooks/envs/tabular/play_bridge_crossing.ipynb",
        "notebooks/envs/mixed/play_cartpole.ipynb",
        "notebooks/envs/tabular/play_colour_bomb_gridworlds.ipynb",
        "notebooks/envs/mixed/play_mountain_car.ipynb",
        "notebooks/envs/continuous/play_obstacles.ipynb",
        "notebooks/envs/discrete/play_pacman_coins.ipynb",
        "notebooks/envs/tabular/play_pacman_tabular.ipynb",
        "notebooks/envs/continuous/play_roads.ipynb",
        "notebooks/envs/discrete/play_safety_gridworlds.ipynb",
    )

    for notebook_path in notebook_paths:
        source = _notebook_source(notebook_path)

        assert "from masa.common.rendering.notebook_play import make_reset_env, sync_selected_env" in source
        assert "follow_selector = env_name is None" in source
        assert "sync_selected_env(" in source
        assert 'print("switched:", env_name)' in source or 'print("switched:", selected_env_name)' in source


def test_all_play_notebooks_offer_opt_in_1080p_recording_in_first_cell():
    notebook_paths = (
        "notebooks/envs/continuous/play_obstacles.ipynb",
        "notebooks/envs/continuous/play_roads.ipynb",
        "notebooks/envs/discrete/play_pacman_coins.ipynb",
        "notebooks/envs/discrete/play_safety_gridworlds.ipynb",
        "notebooks/envs/mixed/play_cartpole.ipynb",
        "notebooks/envs/mixed/play_mountain_car.ipynb",
        "notebooks/envs/multiagent/play_capture_the_flag.ipynb",
        "notebooks/envs/multiagent/play_clean_up.ipynb",
        "notebooks/envs/multiagent/play_markov_stag_hunt.ipynb",
        "notebooks/envs/tabular/play_bridge_crossing.ipynb",
        "notebooks/envs/tabular/play_colour_bomb_gridworlds.ipynb",
        "notebooks/envs/tabular/play_colour_grid_world.ipynb",
        "notebooks/envs/tabular/play_media_streaming.ipynb",
        "notebooks/envs/tabular/play_pacman_tabular.ipynb",
    )

    assert len(notebook_paths) == 14
    for notebook_path in notebook_paths:
        notebook = _load_notebook(notebook_path)
        first_cell = notebook["cells"][0]
        first_source = "".join(first_cell.get("source", []))
        source = _notebook_source(notebook_path)

        assert first_cell["cell_type"] == "code"
        assert first_source.splitlines() == [
            "RECORD_VIDEO = False",
            f'VIDEO_PATH = "videos/notebooks/{Path(notebook_path).stem}.mp4"',
        ]
        assert source.count("RECORD_VIDEO = False") == 1
        assert source.count("VIDEO_PATH = ") == 1
        assert "1920x1080" in source
        assert "notebook_video_recording" in source or "start_recorded_play_thread" in source


def test_roads_notebook_runs_recorded_play_loop_in_background_thread():
    source = _notebook_source("notebooks/envs/continuous/play_roads.ipynb")

    assert "from masa.common.rendering.notebook_play import start_recorded_play_thread" in source
    assert "def _run(stop_event):" in source
    assert "while running and not stop_event.is_set() and not env.human_window_closed:" in source
    assert 'return start_recorded_play_thread("roads", _run, record_video=RECORD_VIDEO, video_path=VIDEO_PATH)' in source
