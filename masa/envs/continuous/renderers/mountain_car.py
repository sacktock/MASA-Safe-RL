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
HILL_COLOR = (98, 133, 99)
HILL_SHADOW_COLOR = (71, 94, 77)
GOAL_COLOR = (92, 166, 111)
WALL_COLOR = (202, 76, 75)
CAR_COLOR = (66, 121, 210)
CAR_SHADOW_COLOR = (39, 67, 126)
WHEEL_COLOR = (39, 45, 58)
WINDOW_COLOR = (185, 215, 229)
TEXT_COLOR = (39, 45, 58)
MUTED_TEXT_COLOR = (95, 103, 111)


class MountainCarEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    action_space: Any
    _state: np.ndarray
    _min_position: float
    _max_position: float
    _max_speed: float
    _goal_position: float
    _power: float
    _last_action: Any
    _step_count: int


class MountainCarRenderer:
    """Renderer for the discrete and continuous MountainCar environments."""

    def __init__(self, env: MountainCarEnv) -> None:
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
        action = "none" if snapshot.last_action is None else str(snapshot.last_action)
        return (
            f"position={snapshot.position:.3f}\n"
            f"velocity={snapshot.velocity:.3f}\n"
            f"status={snapshot.status}\n"
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
            pygame.display.set_caption("MASA - MountainCar")
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

    def _snapshot(self) -> "_MountainCarSnapshot":
        state = np.zeros(2, dtype=np.float32)
        if getattr(self.env, "_state", None) is not None:
            state = np.asarray(self.env._state, dtype=np.float32)
        position, velocity = (float(v) for v in state)
        min_position = float(self.env._min_position)
        max_position = float(self.env._max_position)
        max_speed = float(self.env._max_speed)
        goal_position = float(self.env._goal_position)
        status = _status(position, velocity, min_position, max_speed, goal_position)
        return _MountainCarSnapshot(
            position=position,
            velocity=velocity,
            min_position=min_position,
            max_position=max_position,
            max_speed=max_speed,
            goal_position=goal_position,
            power=float(self.env._power),
            last_action=getattr(self.env, "_last_action", None),
            step_count=int(getattr(self.env, "_step_count", 0)),
            status=status,
        )

    def _draw_scene(self, surface: Any, snapshot: "_MountainCarSnapshot", size: int) -> None:
        import pygame
        from pygame import gfxdraw

        surface.fill(PANEL_COLOR)
        panel = surface.get_rect()
        scale = _gym_scale(snapshot, size)

        hill_points = _hill_points(snapshot, scale, size)
        hill_polygon = hill_points + [(panel.right, panel.bottom), (panel.left, panel.bottom)]
        pygame.draw.polygon(surface, HILL_SHADOW_COLOR, [(x, y + max(2, size // 120)) for x, y in hill_polygon])
        pygame.draw.polygon(surface, HILL_COLOR, hill_polygon)
        pygame.draw.aalines(surface, TEXT_COLOR, False, hill_points)

        wall_width = max(3, size // 120)
        wall_x, wall_y = _screen_point(snapshot.min_position, _hill_height(snapshot.min_position) * scale, snapshot, scale, size)
        wall_x = max(wall_width // 2, wall_x)
        pygame.draw.line(surface, WALL_COLOR, (wall_x, wall_y - size // 14), (wall_x, wall_y + size // 16), width=wall_width)

        goal_x, goal_y = _screen_point(snapshot.goal_position, _hill_height(snapshot.goal_position) * scale, snapshot, scale, size)
        flag_height = max(32, int(size * 50 / 600))
        flag_width = max(16, int(size * 25 / 600))
        flag_tip_offset = max(3, int(size * 5 / 600))
        pygame.draw.line(surface, GOAL_COLOR, (goal_x, goal_y), (goal_x, goal_y - flag_height), width=max(2, size // 180))
        flag = [
            (goal_x, goal_y - flag_height),
            (goal_x, goal_y - flag_height + flag_tip_offset * 2),
            (goal_x + flag_width, goal_y - flag_height + flag_tip_offset),
        ]
        gfxdraw.aapolygon(surface, flag, GOAL_COLOR)
        gfxdraw.filled_polygon(surface, flag, GOAL_COLOR)

        self._draw_car(surface, snapshot, scale, size)

        action_force = _last_action_force(snapshot.last_action, snapshot.power)
        action_label = "last action: none" if action_force is None else f"last force: {action_force:.4f}"
        action_color = MUTED_TEXT_COLOR if action_force is None else GOAL_COLOR if action_force > 0 else WALL_COLOR if action_force < 0 else TEXT_COLOR
        self._draw_text(surface, "MountainCar", (panel.left + size // 28, panel.top + size // 28), max(18, size // 18), TEXT_COLOR)
        self._draw_text(surface, f"x {snapshot.position:.2f}  v {snapshot.velocity:.3f}", (panel.left + size // 28, panel.top + size // 10), max(14, size // 25), MUTED_TEXT_COLOR)
        self._draw_text(surface, snapshot.status, (panel.left + size // 28, panel.top + size // 6), max(16, size // 22), GOAL_COLOR if snapshot.status == "goal" else WALL_COLOR if snapshot.status == "wall" else TEXT_COLOR)
        self._draw_text(surface, action_label, (panel.left + size // 3, panel.top + size // 6), max(16, size // 22), action_color)

    def _draw_car(
        self,
        surface: Any,
        snapshot: "_MountainCarSnapshot",
        scale: float,
        size: int,
    ) -> None:
        import pygame
        from pygame import gfxdraw

        body_points, window_points, wheel_centers, wheel_radius = _car_geometry(snapshot, scale, size)

        shadow_points = [(x + max(2, size // 120), y + max(2, size // 120)) for x, y in body_points]
        pygame.draw.polygon(surface, CAR_SHADOW_COLOR, shadow_points)
        body_color = CAR_COLOR if snapshot.status != "wall" else WALL_COLOR
        gfxdraw.aapolygon(surface, body_points, body_color)
        gfxdraw.filled_polygon(surface, body_points, body_color)
        gfxdraw.aapolygon(surface, window_points, WINDOW_COLOR)
        gfxdraw.filled_polygon(surface, window_points, WINDOW_COLOR)

        for wheel_x, wheel_y in wheel_centers:
            gfxdraw.aacircle(surface, wheel_x, wheel_y, wheel_radius, WHEEL_COLOR)
            gfxdraw.filled_circle(surface, wheel_x, wheel_y, wheel_radius, WHEEL_COLOR)

    def _draw_text(self, surface: Any, text: str, pos: Position, size: int, color: RGBColor) -> None:
        import pygame

        pygame.font.init()
        font = pygame.font.Font(None, int(size))
        rendered = font.render(text, True, color)
        surface.blit(rendered, pos)


class _MountainCarSnapshot:
    def __init__(
        self,
        *,
        position: float,
        velocity: float,
        min_position: float,
        max_position: float,
        max_speed: float,
        goal_position: float,
        power: float,
        last_action: Any,
        step_count: int,
        status: str,
    ) -> None:
        self.position = position
        self.velocity = velocity
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.power = power
        self.last_action = last_action
        self.step_count = step_count
        self.status = status


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


def _hill_height(position: float) -> float:
    return math.sin(3.0 * position) * 0.45 + 0.55


def _gym_scale(snapshot: _MountainCarSnapshot, size: int) -> float:
    return size / (snapshot.max_position - snapshot.min_position)


def _hill_points(
    snapshot: _MountainCarSnapshot,
    scale: float,
    size: int,
) -> list[Position]:
    return [
        _screen_point(float(position), _hill_height(float(position)) * scale, snapshot, scale, size)
        for position in np.linspace(snapshot.min_position, snapshot.max_position, 100)
    ]


def _screen_point(
    position: float,
    y_from_bottom: float,
    snapshot: _MountainCarSnapshot,
    scale: float,
    size: int,
) -> Position:
    x = (position - snapshot.min_position) * scale
    return int(round(x)), int(round(size - y_from_bottom))


def _car_geometry(
    snapshot: _MountainCarSnapshot,
    scale: float,
    size: int,
) -> tuple[list[Position], list[Position], list[Position], int]:
    position = max(snapshot.min_position, min(snapshot.max_position, snapshot.position))
    car_width = max(28.0, size * 40 / 600)
    car_height = max(14.0, size * 20 / 600)
    clearance = max(6.0, size * 10 / 600)
    base_x = (position - snapshot.min_position) * scale
    base_y = clearance + _hill_height(position) * scale
    angle = math.cos(3.0 * position)

    body_points = _local_points_to_screen(
        [(-car_width / 2, 0.0), (-car_width / 2, car_height), (car_width / 2, car_height), (car_width / 2, 0.0)],
        base_x,
        base_y,
        angle,
        size,
    )
    window_points = _local_points_to_screen(
        [
            (-car_width * 0.22, car_height * 0.48),
            (car_width * 0.22, car_height * 0.48),
            (car_width * 0.16, car_height * 0.78),
            (-car_width * 0.16, car_height * 0.78),
        ],
        base_x,
        base_y,
        angle,
        size,
    )
    wheel_centers = _local_points_to_screen(
        [(car_width / 4, 0.0), (-car_width / 4, 0.0)],
        base_x,
        base_y,
        angle,
        size,
    )
    return body_points, window_points, wheel_centers, max(4, int(round(car_height / 2.5)))


def _local_points_to_screen(
    points: list[tuple[float, float]],
    base_x: float,
    base_y: float,
    angle: float,
    size: int,
) -> list[Position]:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return [
        (
            int(round(base_x + x * cos_angle - y * sin_angle)),
            int(round(size - (base_y + x * sin_angle + y * cos_angle))),
        )
        for x, y in points
    ]


def _status(position: float, velocity: float, min_position: float, max_speed: float, goal_position: float) -> str:
    if position >= goal_position:
        return "goal"
    if position <= min_position:
        return "wall"
    if abs(velocity) >= max_speed:
        return "max_speed"
    return "driving"


def _last_action_force(last_action: Any, power: float) -> float | None:
    if last_action is None:
        return None
    try:
        if isinstance(last_action, (int, np.integer)):
            return power if int(last_action) else -power
        value = float(np.asarray(last_action).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return power * value


__all__ = ["MountainCarRenderer", "RGBColor", "validate_renderer_options"]
