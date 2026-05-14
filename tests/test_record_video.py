import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo as GymnasiumRecordVideo
from pettingzoo import ParallelEnv

from masa.common.constraints.multi_agent.cmg import Budget
from masa.common.pettingzoo_record_video import RecordVideoParallel
from masa.common.wrappers import RewardMonitor
from masa.envs.discrete.conveyor_belt import cost_fn as conveyor_cost_fn
from masa.envs.discrete.conveyor_belt import label_fn as conveyor_label_fn


def _always(_):
    return True


def _never(_):
    return False


def test_make_env_records_video(tmp_path):
    from masa.common.utils import make_env

    video_folder = tmp_path / "gymnasium-videos"
    env = make_env(
        "conveyor_belt",
        "cmdp",
        5,
        label_fn=conveyor_label_fn,
        env_kwargs={"render_mode": "rgb_array", "render_window_size": 64},
        record_video=True,
        video_folder=str(video_folder),
        video_kwargs={"episode_trigger": _always, "gc_trigger": _never},
        cost_fn=conveyor_cost_fn,
        budget=10.0,
    )

    assert isinstance(env, GymnasiumRecordVideo)

    try:
        env.reset(seed=0)
        env.step(env.action_space.sample())
    finally:
        env.close()

    videos = list(video_folder.glob("*.mp4"))
    assert videos
    assert all(path.stat().st_size > 0 for path in videos)


def test_make_env_recording_is_off_by_default():
    from masa.common.utils import make_env

    env = make_env(
        "conveyor_belt",
        "cmdp",
        5,
        label_fn=conveyor_label_fn,
        cost_fn=conveyor_cost_fn,
        budget=10.0,
    )

    try:
        assert isinstance(env, RewardMonitor)
        assert not isinstance(env, GymnasiumRecordVideo)
    finally:
        env.close()


class DummyVideoParallelEnv(ParallelEnv):
    metadata = {"name": "dummy_video_parallel_v0", "render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = "rgb_array"):
        if render_mode != "rgb_array":
            raise ValueError("DummyVideoParallelEnv only supports render_mode='rgb_array'.")
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]
        self.agents = []
        self.label_fn = lambda obs: {"unsafe"} if obs[0] else set()
        self.cost_fn = lambda labels: 1.0 if "unsafe" in labels else 0.0
        self._step = 0

    def reset(self, seed=None, options=None):
        del seed, options
        self.agents = self.possible_agents[:]
        self._step = 0
        obs = {agent: np.array([0], dtype=np.int8) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(self, actions):
        del actions
        self._step += 1
        obs = {agent: np.array([1], dtype=np.int8) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: True for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :, 0] = min(self._step * 80, 255)
        frame[:, :, 1] = 64
        frame[:, :, 2] = 128
        return frame

    def close(self):
        self.agents = []

    def observation_space(self, agent):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)

    def action_space(self, agent):
        return spaces.Discrete(2)


def test_make_marl_env_records_video(tmp_path):
    from masa.common.registry import MARL_ENV_REGISTRY
    from masa.common.utils import make_marl_env

    MARL_ENV_REGISTRY._items["dummy_video_parallel"] = DummyVideoParallelEnv

    video_folder = tmp_path / "pettingzoo-videos"
    env = make_marl_env(
        "dummy_video_parallel",
        "cmg",
        env_kwargs={"render_mode": "rgb_array"},
        record_video=True,
        video_folder=str(video_folder),
        video_kwargs={"episode_trigger": _always, "gc_trigger": _never},
        budgets=[Budget(amount=3.0, agents=("player_0", "player_1"), name="shared")],
    )

    assert isinstance(env, RecordVideoParallel)

    try:
        env.reset(seed=0)
        env.step({"player_0": 0, "player_1": 0})
    finally:
        env.close()

    videos = list(video_folder.glob("*.mp4"))
    assert videos
    assert all(path.stat().st_size > 0 for path in videos)


def test_make_marl_env_recording_is_off_by_default():
    from masa.common.utils import make_marl_env

    env = make_marl_env(
        "chicken_matrix",
        "cmg",
        env_kwargs={"max_moves": 1},
        budgets=[Budget(amount=3.0, agents=("player_0", "player_1"), name="shared")],
    )

    try:
        assert not isinstance(env, RecordVideoParallel)
    finally:
        env.close()
