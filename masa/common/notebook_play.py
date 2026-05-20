from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
import threading
from typing import Any


MakeEnv = Callable[..., Any]
ThreadTarget = Callable[[threading.Event], None]
_PLAY_SESSIONS: dict[str, "NotebookPlaySession"] = {}
_PLAY_SESSIONS_LOCK = threading.Lock()


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
    "close_pygame_env",
    "make_reset_env",
    "start_play_thread",
    "stop_play_thread",
    "sync_selected_env",
]
