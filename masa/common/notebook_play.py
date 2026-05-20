from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any


MakeEnv = Callable[..., Any]


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


__all__ = ["close_pygame_env", "make_reset_env", "sync_selected_env"]
