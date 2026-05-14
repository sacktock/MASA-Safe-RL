from __future__ import annotations

import gc
import os
from typing import Any, Callable, TypeAlias

import gymnasium
import numpy as np
from gymnasium.error import DependencyNotInstalled
from pettingzoo.utils.env import AECEnv, ActionType, AgentID, ObsType, ParallelEnv
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

RenderFrame: TypeAlias = np.typing.NDArray[Any]


class RecordVideoParallel(BaseParallelWrapper):
    """Record videos from a PettingZoo parallel environment."""

    def __init__(
        self,
        env: ParallelEnv,
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ):
        super().__init__(env)
        assert isinstance(env, ParallelEnv), "RecordVideoParallel is only compatible with ParallelEnv environments."

        if env.render_mode in {None, "human", "ansi"}:  # type: ignore[attr-defined]
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideoParallel. "  # type: ignore[attr-defined]
                "Initialize your environment with a render_mode that returns an image, such as rgb_array."
            )

        if episode_trigger is None and step_trigger is None:
            episode_trigger = (
                lambda episode_id: int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
                if episode_id < 1000
                else episode_id % 1000 == 0
            )

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger
        self.gc_trigger = gc_trigger

        self.video_folder = os.path.abspath(video_folder)
        if os.path.isdir(self.video_folder):
            gymnasium.logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                "(try specifying a different `video_folder` for the `RecordVideoParallel` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        if fps is None:
            fps = int(getattr(env, "metadata", {}).get("render_fps", 30))
        self.frames_per_sec: int = fps
        self.name_prefix: str = name_prefix
        self._video_name: str | None = None
        self.video_length: int | float = video_length if video_length != 0 else float("inf")
        self.recording: bool = False
        self.recorded_frames: list[RenderFrame] = []
        self.render_history: list[RenderFrame] = []

        self.step_id: int = -1
        self.episode_id: int = -1

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise DependencyNotInstalled(
                'MoviePy is not installed, run `pip install "moviepy>=2.2.1,<3.0.0"`'
            ) from e

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()
        if isinstance(frame, list):
            if len(frame) == 0:
                return
            self.render_history += frame
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()
            gymnasium.logger.warn(
                f"Recording stopped: expected type of frame returned by render to be a numpy array, got {type(frame)}."
            )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.video_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return obs, info

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, rew, terminated, truncated, info = self.env.step(actions)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return obs, rew, terminated, truncated, info

    def render(self):
        render_out = self.env.render()
        if self.recording and isinstance(render_out, list):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            tmp_history = self.render_history
            self.render_history = []
            frames = render_out if isinstance(render_out, list) else [render_out]
            return tmp_history + frames
        return render_out

    def close(self):
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, video_name: str):
        if self.recording:
            self.stop_recording()

        self.recording = True
        self._video_name = video_name

    def stop_recording(self):
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            gymnasium.logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "moviepy>=2.2.1,<3.0.0"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(path, logger=moviepy_logger)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()

    def __del__(self):
        if len(getattr(self, "recorded_frames", [])) > 0:
            gymnasium.logger.warn("Unable to save last video! Did you call close()?")


class RecordVideoAEC(BaseWrapper):
    """Record videos from a PettingZoo AEC environment."""

    def __init__(
        self,
        env: AECEnv,
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ):
        super().__init__(env)
        assert isinstance(env, AECEnv), "RecordVideoAEC is only compatible with AECEnv environments."

        if env.render_mode in {None, "human", "ansi"}:  # type: ignore[attr-defined]
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideoAEC. "  # type: ignore[attr-defined]
                "Initialize your environment with a render_mode that returns an image, such as rgb_array."
            )

        if episode_trigger is None and step_trigger is None:
            episode_trigger = (
                lambda episode_id: int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
                if episode_id < 1000
                else episode_id % 1000 == 0
            )

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger
        self.gc_trigger = gc_trigger

        self.video_folder = os.path.abspath(video_folder)
        if os.path.isdir(self.video_folder):
            gymnasium.logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                "(try specifying a different `video_folder` for the `RecordVideoAEC` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        if fps is None:
            fps = int(getattr(env, "metadata", {}).get("render_fps", 30))
        self.frames_per_sec: int = fps
        self.name_prefix: str = name_prefix
        self._video_name: str | None = None
        self.video_length: int | float = video_length if video_length != 0 else float("inf")
        self.recording: bool = False
        self.recorded_frames: list[RenderFrame] = []
        self.render_history: list[RenderFrame] = []

        self.step_id: int = -1
        self.episode_id: int = -1

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise DependencyNotInstalled(
                'MoviePy is not installed, run `pip install "moviepy>=2.2.1,<3.0.0"`'
            ) from e

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()
        if isinstance(frame, list):
            if len(frame) == 0:
                return
            self.render_history += frame
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()
            gymnasium.logger.warn(
                f"Recording stopped: expected type of frame returned by render to be a numpy array, got {type(frame)}."
            )

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.video_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

    def step(self, action: ActionType):
        self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

    def render(self):
        render_out = self.env.render()
        if self.recording and isinstance(render_out, list):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            tmp_history = self.render_history
            self.render_history = []
            frames = render_out if isinstance(render_out, list) else [render_out]
            return tmp_history + frames
        return render_out

    def close(self):
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, video_name: str):
        if self.recording:
            self.stop_recording()

        self.recording = True
        self._video_name = video_name

    def stop_recording(self):
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            gymnasium.logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "moviepy>=2.2.1,<3.0.0"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(path, logger=moviepy_logger)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()

    def __del__(self):
        if len(getattr(self, "recorded_frames", [])) > 0:
            gymnasium.logger.warn("Unable to save last video! Did you call close()?")


RecordVideo = RecordVideoParallel
