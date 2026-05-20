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

        surface.fill(PANEL_COLOR)
        panel = surface.get_rect()

        track_left = panel.left + max(18, panel.width // 16)
        track_right = panel.right - max(18, panel.width // 16)
        track_top = panel.top + int(panel.height * 0.26)
        track_bottom = panel.bottom - int(panel.height * 0.22)

        hill_points = _hill_points(snapshot, track_left, track_right, track_top, track_bottom)
        hill_polygon = hill_points + [(track_right, panel.bottom), (track_left, panel.bottom)]
        pygame.draw.polygon(surface, HILL_SHADOW_COLOR, [(x, y + max(2, size // 120)) for x, y in hill_polygon])
        pygame.draw.polygon(surface, HILL_COLOR, hill_polygon)
        pygame.draw.lines(surface, TEXT_COLOR, False, hill_points, width=max(2, size // 150))

        wall_x, wall_y = _point_for_position(snapshot.min_position, snapshot, track_left, track_right, track_top, track_bottom)
        pygame.draw.line(surface, WALL_COLOR, (wall_x, wall_y - size // 14), (wall_x, wall_y + size // 16), width=max(3, size // 120))

        goal_x, goal_y = _point_for_position(snapshot.goal_position, snapshot, track_left, track_right, track_top, track_bottom)
        flag_height = max(42, size // 6)
        pygame.draw.line(surface, GOAL_COLOR, (goal_x, goal_y), (goal_x, goal_y - flag_height), width=max(3, size // 120))
        flag = [
            (goal_x, goal_y - flag_height),
            (goal_x + max(26, size // 10), goal_y - flag_height + max(9, size // 36)),
            (goal_x, goal_y - flag_height + max(18, size // 18)),
        ]
        pygame.draw.polygon(surface, GOAL_COLOR, flag)

        car_x, car_y = _point_for_position(snapshot.position, snapshot, track_left, track_right, track_top, track_bottom)
        car_angle = _slope_angle(snapshot.position, snapshot, track_left, track_right, track_top, track_bottom)
        self._draw_car(surface, (car_x, car_y), car_angle, snapshot, size)

        action_force = _last_action_force(snapshot.last_action, snapshot.power)
        action_label = "last action: none" if action_force is None else f"last force: {action_force:.4f}"
        action_color = MUTED_TEXT_COLOR if action_force is None else GOAL_COLOR if action_force > 0 else WALL_COLOR if action_force < 0 else TEXT_COLOR
        self._draw_text(surface, "MountainCar", (panel.left + size // 28, panel.top + size // 28), max(18, size // 18), TEXT_COLOR)
        self._draw_text(surface, f"x {snapshot.position:.2f}  v {snapshot.velocity:.3f}", (panel.left + size // 28, panel.top + size // 10), max(14, size // 25), MUTED_TEXT_COLOR)
        self._draw_text(surface, snapshot.status, (panel.left + size // 28, panel.bottom - size // 12), max(16, size // 22), GOAL_COLOR if snapshot.status == "goal" else WALL_COLOR if snapshot.status == "wall" else TEXT_COLOR)
        self._draw_text(surface, action_label, (panel.left + size // 3, panel.bottom - size // 12), max(16, size // 22), action_color)

    def _draw_car(
        self,
        surface: Any,
        ground: Position,
        angle: float,
        snapshot: "_MountainCarSnapshot",
        size: int,
    ) -> None:
        import pygame

        car_width = max(38, size // 7)
        car_height = max(22, size // 20)
        wheel_radius = max(5, size // 60)
        up_offset = car_height // 2 + wheel_radius
        center = _rotated_offset(ground, 0.0, -up_offset, angle)

        body_points = _rotated_rect(center, car_width, car_height, angle)
        shadow_points = [(x + max(2, size // 120), y + max(2, size // 120)) for x, y in body_points]
        pygame.draw.polygon(surface, CAR_SHADOW_COLOR, shadow_points)
        pygame.draw.polygon(surface, CAR_COLOR if snapshot.status != "wall" else WALL_COLOR, body_points)

        window_center = _rotated_offset(center, 0.0, -car_height * 0.12, angle)
        window_points = _rotated_rect(window_center, car_width * 0.36, car_height * 0.45, angle)
        pygame.draw.polygon(surface, WINDOW_COLOR, window_points)

        wheel_axis_offset = car_width * 0.28
        wheel_y_offset = car_height * 0.48
        for side in (-1, 1):
            wheel_center = _rotated_offset(center, wheel_axis_offset * side, wheel_y_offset, angle)
            pygame.draw.circle(surface, WHEEL_COLOR, wheel_center, wheel_radius)

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


def _hill_bounds(snapshot: _MountainCarSnapshot) -> tuple[float, float]:
    positions = np.linspace(snapshot.min_position, snapshot.max_position, 120)
    heights = [_hill_height(float(position)) for position in positions]
    return min(heights), max(heights)


def _point_for_position(
    position: float,
    snapshot: _MountainCarSnapshot,
    track_left: int,
    track_right: int,
    track_top: int,
    track_bottom: int,
) -> Position:
    min_height, max_height = _hill_bounds(snapshot)
    clipped = max(snapshot.min_position, min(snapshot.max_position, position))
    x_ratio = (clipped - snapshot.min_position) / (snapshot.max_position - snapshot.min_position)
    height_ratio = (_hill_height(clipped) - min_height) / (max_height - min_height)
    x = int(track_left + x_ratio * (track_right - track_left))
    y = int(track_bottom - height_ratio * (track_bottom - track_top))
    return x, y


def _hill_points(
    snapshot: _MountainCarSnapshot,
    track_left: int,
    track_right: int,
    track_top: int,
    track_bottom: int,
) -> list[Position]:
    return [
        _point_for_position(position, snapshot, track_left, track_right, track_top, track_bottom)
        for position in np.linspace(snapshot.min_position, snapshot.max_position, 96)
    ]


def _slope_angle(
    position: float,
    snapshot: _MountainCarSnapshot,
    track_left: int,
    track_right: int,
    track_top: int,
    track_bottom: int,
) -> float:
    delta = 0.01
    left = _point_for_position(position - delta, snapshot, track_left, track_right, track_top, track_bottom)
    right = _point_for_position(position + delta, snapshot, track_left, track_right, track_top, track_bottom)
    return math.atan2(right[1] - left[1], right[0] - left[0])


def _rotated_rect(center: Position, width: float, height: float, angle: float) -> list[Position]:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    corners = [(-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)]
    return [
        (
            int(center[0] + x * cos_angle - y * sin_angle),
            int(center[1] + x * sin_angle + y * cos_angle),
        )
        for x, y in corners
    ]


def _rotated_offset(origin: Position, along: float, down: float, angle: float) -> Position:
    return _offset_point(
        origin,
        math.cos(angle) * along - math.sin(angle) * down,
        math.sin(angle) * along + math.cos(angle) * down,
    )


def _offset_point(origin: Position, dx: float, dy: float) -> Position:
    return int(origin[0] + dx), int(origin[1] + dy)


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
