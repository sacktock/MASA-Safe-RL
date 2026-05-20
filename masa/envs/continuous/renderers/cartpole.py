from __future__ import annotations

import math
import os
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
RGBColor = tuple[int, int, int]

BACKGROUND_COLOR = (237, 238, 232)
PANEL_COLOR = (226, 230, 224)
TRACK_COLOR = (72, 82, 96)
SAFE_COLOR = (92, 166, 111)
DANGER_COLOR = (202, 76, 75)
CART_COLOR = (66, 121, 210)
CART_SHADOW_COLOR = (39, 67, 126)
WHEEL_COLOR = (39, 45, 58)
POLE_COLOR = (198, 126, 63)
POLE_TIP_COLOR = (235, 187, 86)
TEXT_COLOR = (39, 45, 58)
MUTED_TEXT_COLOR = (95, 103, 111)


class CartPoleEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    action_space: Any
    _state: np.ndarray
    _x_threshold: float
    _theta_threshold_radians: float
    _force_mag: float
    _last_action: Any
    _step_count: int


class CartPoleRenderer:
    """Renderer for the discrete and continuous CartPole environments."""

    def __init__(self, env: CartPoleEnv) -> None:
        self.env = env
        self._human_window = None
        self._human_clock = None
        self._human_window_size: Position | None = None
        self._human_window_closed = False

    @property
    def human_window_closed(self) -> bool:
        return self._human_window_closed

    def render(self) -> str | np.ndarray | None:
        mode = self.env.render_mode
        if mode is None:
            return None
        if mode == "ansi":
            return self._render_ansi()
        if mode == "human":
            self._render_human()
            return None
        return self._render_rgb_array()

    def close(self) -> None:
        if self._human_window is not None:
            import pygame

            pygame.display.quit()
            self._human_window = None
            self._human_clock = None
            self._human_window_closed = True

    def handle_pygame_event(self, event: Any) -> bool:
        import pygame

        if event.type == pygame.QUIT:
            self.close()
            return False
        if event.type == pygame.VIDEORESIZE:
            self._human_window_size = (max(160, int(event.w)), max(160, int(event.h)))
        return True

    def _render_ansi(self) -> str:
        snapshot = self._snapshot()
        stable = "stable" if snapshot.stable else "unstable"
        action = "none" if snapshot.last_action is None else str(snapshot.last_action)
        return (
            f"cart x={snapshot.x:.3f} x_dot={snapshot.x_dot:.3f}\n"
            f"pole theta={snapshot.theta:.3f} theta_dot={snapshot.theta_dot:.3f}\n"
            f"status={stable}\n"
            f"last_action={action}"
        )

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        snapshot = self._snapshot()
        size = int(self.env.render_window_size)
        scale = 3
        high_size = size * scale
        surface = pygame.Surface((high_size, high_size))
        surface.fill(BACKGROUND_COLOR)
        self._draw_scene(surface, snapshot, high_size)

        if scale > 1:
            surface = pygame.transform.smoothscale(surface, (size, size))

        frame = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _render_human(self) -> None:
        import pygame

        if self._human_window_closed:
            return

        frame = self._render_rgb_array()
        height, width = frame.shape[:2]
        if self._human_window_size is None:
            self._human_window_size = (width, height)

        if self._human_window is None:
            pygame.init()
            pygame.display.set_caption("MASA - CartPole")
            self._human_window = pygame.display.set_mode(self._human_window_size, pygame.RESIZABLE)
            self._human_clock = pygame.time.Clock()

        for event in pygame.event.get():
            self.handle_pygame_event(event)

        if self._human_window is None or self._human_window_closed:
            return

        window_width, window_height = self._human_window.get_size()
        target_width, target_height = _fit_size((width, height), (window_width, window_height))
        left = (window_width - target_width) // 2
        top = (window_height - target_height) // 2

        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        if (target_width, target_height) != (width, height):
            frame_surface = pygame.transform.smoothscale(frame_surface, (target_width, target_height))

        self._human_window.fill(BACKGROUND_COLOR)
        self._human_window.blit(frame_surface, (left, top))
        pygame.display.flip()
        if self._human_clock is not None:
            self._human_clock.tick(self.env.metadata["render_fps"])

    def _snapshot(self) -> "_CartPoleSnapshot":
        state = np.zeros(4, dtype=np.float32)
        if getattr(self.env, "_state", None) is not None:
            state = np.asarray(self.env._state, dtype=np.float32)
        x, x_dot, theta, theta_dot = (float(v) for v in state)
        stable = abs(theta) <= float(self.env._theta_threshold_radians) and abs(x) <= float(self.env._x_threshold)
        return _CartPoleSnapshot(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            x_threshold=float(self.env._x_threshold),
            theta_threshold=float(self.env._theta_threshold_radians),
            force_mag=float(self.env._force_mag),
            last_action=getattr(self.env, "_last_action", None),
            step_count=int(getattr(self.env, "_step_count", 0)),
            stable=stable,
        )

    def _draw_scene(self, surface: Any, snapshot: "_CartPoleSnapshot", size: int) -> None:
        import pygame

        margin = max(24, size // 14)
        panel = pygame.Rect(margin, margin, size - margin * 2, size - margin * 2)
        pygame.draw.rect(surface, PANEL_COLOR, panel, border_radius=max(8, size // 32))

        track_y = panel.top + int(panel.height * 0.70)
        track_left = panel.left + max(18, panel.width // 14)
        track_right = panel.right - max(18, panel.width // 14)
        track_width = track_right - track_left
        track_rect = pygame.Rect(track_left, track_y - max(4, size // 110), track_width, max(8, size // 55))
        pygame.draw.rect(surface, TRACK_COLOR, track_rect, border_radius=max(3, size // 80))

        safe_left = _x_to_px(-snapshot.x_threshold, snapshot.x_threshold, track_left, track_right)
        safe_right = _x_to_px(snapshot.x_threshold, snapshot.x_threshold, track_left, track_right)
        pygame.draw.line(surface, SAFE_COLOR, (safe_left, track_y - size // 12), (safe_left, track_y + size // 12), width=max(2, size // 120))
        pygame.draw.line(surface, SAFE_COLOR, (safe_right, track_y - size // 12), (safe_right, track_y + size // 12), width=max(2, size // 120))

        cart_x = _x_to_px(snapshot.x, snapshot.x_threshold, track_left, track_right)
        cart_width = max(34, size // 7)
        cart_height = max(18, size // 18)
        cart = pygame.Rect(cart_x - cart_width // 2, track_y - cart_height - max(6, size // 90), cart_width, cart_height)
        pygame.draw.rect(surface, CART_SHADOW_COLOR, cart.move(max(2, size // 120), max(2, size // 120)), border_radius=max(4, size // 70))
        pygame.draw.rect(surface, CART_COLOR if snapshot.stable else DANGER_COLOR, cart, border_radius=max(4, size // 70))

        wheel_radius = max(5, size // 55)
        for wheel_x in (cart.left + cart_width // 4, cart.right - cart_width // 4):
            pygame.draw.circle(surface, WHEEL_COLOR, (wheel_x, cart.bottom + wheel_radius), wheel_radius)

        pivot = (cart.centerx, cart.top)
        pole_len = max(72, int(size * 0.34))
        pole_end = (
            int(pivot[0] + math.sin(snapshot.theta) * pole_len),
            int(pivot[1] - math.cos(snapshot.theta) * pole_len),
        )
        for threshold in (-snapshot.theta_threshold, snapshot.theta_threshold):
            end = (
                int(pivot[0] + math.sin(threshold) * pole_len),
                int(pivot[1] - math.cos(threshold) * pole_len),
            )
            pygame.draw.line(surface, DANGER_COLOR, pivot, end, width=max(2, size // 160))

        pygame.draw.line(surface, POLE_COLOR, pivot, pole_end, width=max(6, size // 38))
        pygame.draw.circle(surface, POLE_TIP_COLOR, pole_end, max(7, size // 45))
        pygame.draw.circle(surface, WHEEL_COLOR, pivot, max(5, size // 60))

        action_force = _last_action_force(snapshot.last_action, snapshot.force_mag)
        action_color = MUTED_TEXT_COLOR if action_force is None else SAFE_COLOR if action_force < 0 else DANGER_COLOR if action_force > 0 else TEXT_COLOR
        action_label = "last action: none" if action_force is None else f"last force: {action_force:.1f}"
        status = "stable" if snapshot.stable else "unstable"
        self._draw_text(surface, "CartPole", (panel.left + size // 28, panel.top + size // 28), max(18, size // 18), TEXT_COLOR)
        self._draw_text(surface, f"x {snapshot.x:.2f}  theta {snapshot.theta:.2f}", (panel.left + size // 28, panel.top + size // 10), max(14, size // 25), MUTED_TEXT_COLOR)
        self._draw_text(surface, status, (panel.left + size // 28, panel.bottom - size // 12), max(16, size // 22), SAFE_COLOR if snapshot.stable else DANGER_COLOR)
        self._draw_text(surface, action_label, (panel.left + size // 3, panel.bottom - size // 12), max(16, size // 22), action_color)

    def _draw_text(self, surface: Any, text: str, pos: Position, size: int, color: RGBColor) -> None:
        import pygame

        pygame.font.init()
        font = pygame.font.Font(None, int(size))
        rendered = font.render(text, True, color)
        surface.blit(rendered, pos)


class _CartPoleSnapshot:
    def __init__(
        self,
        *,
        x: float,
        x_dot: float,
        theta: float,
        theta_dot: float,
        x_threshold: float,
        theta_threshold: float,
        force_mag: float,
        last_action: Any,
        step_count: int,
        stable: bool,
    ) -> None:
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot
        self.x_threshold = x_threshold
        self.theta_threshold = theta_threshold
        self.force_mag = force_mag
        self.last_action = last_action
        self.step_count = step_count
        self.stable = stable


def validate_renderer_options(render_mode: str | None, render_window_size: int) -> None:
    if render_mode not in (None, "ansi", "rgb_array", "human"):
        raise ValueError("render_mode must be None, 'ansi', 'rgb_array', or 'human'.")
    if int(render_window_size) <= 0:
        raise ValueError("render_window_size must be positive.")


def _fit_size(source: Position, target: Position) -> Position:
    source_width, source_height = source
    target_width, target_height = target
    scale = min(target_width / source_width, target_height / source_height)
    return max(1, int(source_width * scale)), max(1, int(source_height * scale))


def _x_to_px(x: float, x_threshold: float, track_left: int, track_right: int) -> int:
    extent = x_threshold * 1.25
    clipped = max(-extent, min(extent, x))
    ratio = (clipped + extent) / (2.0 * extent)
    return int(track_left + ratio * (track_right - track_left))


def _last_action_force(last_action: Any, force_mag: float) -> float | None:
    if last_action is None:
        return None
    try:
        if isinstance(last_action, (int, np.integer)):
            return force_mag if int(last_action) else -force_mag
        value = float(np.asarray(last_action).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return force_mag * value


__all__ = ["CartPoleRenderer", "RGBColor", "validate_renderer_options"]
