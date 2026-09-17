from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Iterator


MakeEnv = Callable[..., Any]
ThreadTarget = Callable[[threading.Event], None]
_PLAY_SESSIONS: dict[str, "NotebookPlaySession"] = {}
_PLAY_SESSIONS_LOCK = threading.Lock()
_VIDEO_RECORDER_LOCK = threading.Lock()
_ACTIVE_VIDEO_RECORDER: "NotebookVideoRecorder | None" = None


@dataclass
class NotebookPlaySession:
    """Handle for a background notebook play loop."""

    stop_event: threading.Event
    thread: threading.Thread

    def stop(self, timeout: float | None = 2.0) -> None:
        self.stop_event.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()


class NotebookVideoRecorder:
    """Record a notebook pygame play session to a fixed-size MP4.

    The recorder hooks pygame's display presentation calls, so it captures only
    the environment canvas (never notebook/desktop chrome). The source aspect
    ratio is preserved and letterboxed onto ``output_size``.

    Recording is deliberately disabled unless ``enabled=True``.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        enabled: bool = False,
        output_size: tuple[int, int] = (1920, 1080),
        fps: int = 30,
        background: tuple[int, int, int] = (12, 12, 12),
        max_idle_gap: float = 1.0,
    ) -> None:
        width, height = (int(output_size[0]), int(output_size[1]))
        if width < 1 or height < 1:
            raise ValueError("output_size dimensions must be positive")
        if fps < 1:
            raise ValueError("fps must be at least 1")
        if max_idle_gap <= 0:
            raise ValueError("max_idle_gap must be positive")

        self.output_path = Path(output_path)
        self.enabled = bool(enabled)
        self.output_size = (width, height)
        self.fps = int(fps)
        self.background = tuple(int(channel) for channel in background)
        self.max_idle_gap = float(max_idle_gap)

        self._pygame: Any | None = None
        self._writer: Any | None = None
        self._original_flip: Any | None = None
        self._original_update: Any | None = None
        self._next_frame_time: float | None = None
        self._installed = False

    def __enter__(self) -> "NotebookVideoRecorder":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def install(self) -> None:
        """Install pygame display hooks when recording is enabled."""
        global _ACTIVE_VIDEO_RECORDER

        if not self.enabled or self._installed:
            return

        import pygame

        with _VIDEO_RECORDER_LOCK:
            if _ACTIVE_VIDEO_RECORDER is not None and _ACTIVE_VIDEO_RECORDER is not self:
                raise RuntimeError("another notebook video recorder is already active")
            _ACTIVE_VIDEO_RECORDER = self

        self._pygame = pygame
        self._original_flip = pygame.display.flip
        self._original_update = pygame.display.update

        def flip_and_record(*args, **kwargs):
            result = self._original_flip(*args, **kwargs)
            self.capture()
            return result

        def update_and_record(*args, **kwargs):
            result = self._original_update(*args, **kwargs)
            self.capture()
            return result

        pygame.display.flip = flip_and_record
        pygame.display.update = update_and_record
        self._installed = True

    def _ensure_writer(self) -> Any:
        if self._writer is not None:
            return self._writer

        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = FFMPEG_VideoWriter(
            str(self.output_path),
            self.output_size,
            self.fps,
            codec="libx264",
            preset="medium",
            ffmpeg_params=["-crf", "18", "-movflags", "+faststart"],
            pixel_format="yuv420p",
        )
        return self._writer

    def _current_frame(self):
        import numpy as np

        pygame = self._pygame
        if pygame is None:
            return None
        surface = pygame.display.get_surface()
        if surface is None:
            return None

        source_width, source_height = surface.get_size()
        if source_width < 1 or source_height < 1:
            return None

        output_width, output_height = self.output_size
        scale = min(output_width / source_width, output_height / source_height)
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))

        canvas = pygame.Surface(self.output_size)
        canvas.fill(self.background)
        if (target_width, target_height) == (source_width, source_height):
            scaled = surface
        else:
            scaled = pygame.transform.smoothscale(surface, (target_width, target_height))

        left = (output_width - target_width) // 2
        top = (output_height - target_height) // 2
        canvas.blit(scaled, (left, top))

        frame = np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def capture(self) -> None:
        """Capture the current pygame display while preserving real-time pacing."""
        if not self.enabled or not self._installed:
            return

        frame = self._current_frame()
        if frame is None:
            return

        now = time.monotonic()
        frame_period = 1.0 / self.fps

        if self._next_frame_time is None:
            self._next_frame_time = now
        elif now - self._next_frame_time > self.max_idle_gap:
            # Do not turn a long pause between user inputs into many seconds of
            # duplicate frames in a demo clip.
            self._next_frame_time = now

        writer = self._ensure_writer()
        while self._next_frame_time <= now + frame_period * 0.25:
            writer.write_frame(frame)
            self._next_frame_time += frame_period

    def close(self) -> None:
        """Restore pygame hooks and finalise the MP4, if one was started."""
        global _ACTIVE_VIDEO_RECORDER

        if self._installed and self._pygame is not None:
            with suppress(Exception):
                if self._original_flip is not None:
                    self._pygame.display.flip = self._original_flip
                if self._original_update is not None:
                    self._pygame.display.update = self._original_update
            self._installed = False

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        with _VIDEO_RECORDER_LOCK:
            if _ACTIVE_VIDEO_RECORDER is self:
                _ACTIVE_VIDEO_RECORDER = None


@contextmanager
def notebook_video_recording(
    enabled: bool,
    output_path: str | Path,
    *,
    output_size: tuple[int, int] = (1920, 1080),
    fps: int = 30,
) -> Iterator[NotebookVideoRecorder]:
    """Context manager for opt-in 1080p recording of notebook play sessions."""
    recorder = NotebookVideoRecorder(
        output_path,
        enabled=enabled,
        output_size=output_size,
        fps=fps,
    )
    with recorder:
        yield recorder


def stop_play_thread(session_key: str, *, timeout: float | None = 2.0) -> None:
    """Stop a previously started notebook play thread, if one exists."""
    with _PLAY_SESSIONS_LOCK:
        session = _PLAY_SESSIONS.pop(session_key, None)
    if session is not None:
        session.stop(timeout=timeout)


def start_play_thread(session_key: str, target: ThreadTarget) -> NotebookPlaySession:
    """Start a managed notebook play thread, replacing any existing session."""
    stop_play_thread(session_key)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=target,
        args=(stop_event,),
        daemon=True,
        name=f"notebook-play-{session_key}",
    )
    session = NotebookPlaySession(stop_event=stop_event, thread=thread)
    with _PLAY_SESSIONS_LOCK:
        _PLAY_SESSIONS[session_key] = session
    thread.start()
    return session


def start_recorded_play_thread(
    session_key: str,
    target: ThreadTarget,
    *,
    record_video: bool = False,
    video_path: str | Path,
    video_size: tuple[int, int] = (1920, 1080),
    video_fps: int = 30,
) -> NotebookPlaySession:
    """Start a managed play thread with optional notebook video recording."""

    def recorded_target(stop_event: threading.Event) -> None:
        with notebook_video_recording(
            record_video,
            video_path,
            output_size=video_size,
            fps=video_fps,
        ):
            target(stop_event)

    return start_play_thread(session_key, recorded_target)


def close_pygame_env(env: Any, *, pygame: Any | None = None) -> None:
    """Close a notebook pygame environment and release the display if available."""
    if pygame is not None:
        with suppress(Exception):
            pygame.event.clear()
    env.close()
    if pygame is None:
        return
    with suppress(Exception):
        if pygame.display.get_init():
            pygame.display.quit()


def make_reset_env(
    make_env: MakeEnv,
    env_name: str,
    *,
    seed: int | None,
    render_mode: str = "human",
    render_window_size: int,
    env_kwargs: Mapping[str, Any] | None = None,
):
    kwargs = dict(env_kwargs or {})
    env = make_env(
        env_name,
        render_mode=render_mode,
        render_window_size=render_window_size,
        **kwargs,
    )
    obs, info = env.reset(seed=seed)
    return env, obs, info


def sync_selected_env(
    env: Any,
    env_name: str,
    selector: Any,
    make_env: MakeEnv,
    *,
    seed: int | None,
    render_mode: str = "human",
    render_window_size: int,
    env_kwargs: Mapping[str, Any] | None = None,
    pygame: Any | None = None,
):
    selected_env_name = selector.value
    if selected_env_name == env_name:
        return env, env_name, None, None, False

    close_pygame_env(env, pygame=pygame)
    env, obs, info = make_reset_env(
        make_env,
        selected_env_name,
        seed=seed,
        render_mode=render_mode,
        render_window_size=render_window_size,
        env_kwargs=env_kwargs,
    )
    return env, selected_env_name, obs, info, True


__all__ = [
    "NotebookPlaySession",
    "NotebookVideoRecorder",
    "close_pygame_env",
    "make_reset_env",
    "notebook_video_recording",
    "start_play_thread",
    "start_recorded_play_thread",
    "stop_play_thread",
    "sync_selected_env",
]
