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
ROAD_COLOR = (239, 241, 235)
GRID_COLOR = (199, 205, 196)
GOAL_COLOR = (92, 166, 111)
GOAL_FILL_COLOR = (202, 225, 204)
BOUNDARY_COLOR = (72, 82, 96)
AGENT_COLOR = (66, 121, 210)
AGENT_SHADOW_COLOR = (39, 67, 126)
VELOCITY_COLOR = (198, 126, 63)
ACTION_COLOR = (202, 76, 75)
SPEED_LIMIT_COLOR = (235, 187, 86)
TEXT_COLOR = (39, 45, 58)
MUTED_TEXT_COLOR = (95, 103, 111)


class Road1DEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    action_space: Any
    _state: np.ndarray
    _min_position: float
    _max_position: float
    _max_speed: float
    _goal_position: float
    _speed_limit: float
    _last_action: Any
    _step_count: int


class Road1DRenderer:
    """Renderer for the continuous Road1D environment."""

    def __init__(self, env: Road1DEnv) -> None:
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
            pygame.display.set_caption("MASA - Road1D")
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

    def _snapshot(self) -> "_Road1DSnapshot":
        state = np.zeros(2, dtype=np.float32)
        if getattr(self.env, "_state", None) is not None:
            state = np.asarray(self.env._state, dtype=np.float32)
        position = float(state[0])
        velocity = float(state[1])
        min_position = float(self.env._min_position)
        max_position = float(self.env._max_position)
        max_speed = float(self.env._max_speed)
        speed_limit = float(self.env._speed_limit)
        goal_position = float(self.env._goal_position)
        status = _status(position, velocity, min_position, max_position, max_speed, speed_limit, goal_position)
        return _Road1DSnapshot(
            position=position,
            velocity=velocity,
            min_position=min_position,
            max_position=max_position,
            max_speed=max_speed,
            speed_limit=speed_limit,
            goal_position=goal_position,
            last_action=getattr(self.env, "_last_action", None),
            step_count=int(getattr(self.env, "_step_count", 0)),
            status=status,
        )

    def _draw_scene(self, surface: Any, snapshot: "_Road1DSnapshot", size: int) -> None:
        import pygame

        margin = max(24, size // 14)
        panel = pygame.Rect(margin, margin, size - margin * 2, size - margin * 2)
        pygame.draw.rect(surface, PANEL_COLOR, panel, border_radius=max(8, size // 32))

        track_left = panel.left + max(28, panel.width // 12)
        track_right = panel.right - max(28, panel.width // 12)
        track_y = panel.top + int(panel.height * 0.58)
        track_height = max(28, size // 12)
        track_rect = pygame.Rect(track_left, track_y - track_height // 2, track_right - track_left, track_height)

        pygame.draw.rect(surface, ROAD_COLOR, track_rect, border_radius=max(6, size // 70))
        pygame.draw.rect(surface, BOUNDARY_COLOR, track_rect, width=max(3, size // 110), border_radius=max(6, size // 70))

        goal_x = _x_to_px(snapshot.goal_position, snapshot, track_left, track_right)
        goal_rect = pygame.Rect(goal_x, track_rect.top, track_right - goal_x, track_rect.height)
        pygame.draw.rect(surface, GOAL_FILL_COLOR, goal_rect, border_radius=max(4, size // 90))
        pygame.draw.line(surface, GOAL_COLOR, (goal_x, track_rect.top - size // 18), (goal_x, track_rect.bottom + size // 18), width=max(3, size // 120))

        for value in range(math.ceil(snapshot.min_position), math.floor(snapshot.max_position) + 1):
            tick_x = _x_to_px(float(value), snapshot, track_left, track_right)
            tick_height = size // 13 if value == 0 else size // 22
            pygame.draw.line(surface, MUTED_TEXT_COLOR if value == 0 else GRID_COLOR, (tick_x, track_y - tick_height), (tick_x, track_y + tick_height), width=max(1, size // 180))

        agent_x = _x_to_px(snapshot.position, snapshot, track_left, track_right)
        agent = (agent_x, track_y)
        velocity_vector = np.array([_scalar_to_screen(snapshot.velocity, snapshot.max_speed, track_rect.width * 0.18), 0.0], dtype=np.float32)
        self._draw_arrow(surface, agent, velocity_vector, VELOCITY_COLOR, size)

        action_value = _last_action_value(snapshot.last_action)
        if action_value is not None:
            action_vector = np.array([_scalar_to_screen(action_value, 2.0, track_rect.width * 0.12), 0.0], dtype=np.float32)
            self._draw_arrow(surface, agent, action_vector, ACTION_COLOR, size)

        agent_radius = max(9, size // 38)
        pygame.draw.circle(surface, AGENT_SHADOW_COLOR, (agent[0] + max(2, size // 120), agent[1] + max(2, size // 120)), agent_radius)
        pygame.draw.circle(surface, _status_color(snapshot.status), agent, agent_radius)
        pygame.draw.circle(surface, TEXT_COLOR, agent, agent_radius, width=max(2, size // 150))

        action_label = "last action: none" if action_value is None else f"last action: {action_value:.2f}"
        self._draw_text(surface, "Road1D", (panel.left + size // 28, panel.top + size // 28), max(18, size // 18), TEXT_COLOR)
        self._draw_text(surface, f"x {snapshot.position:.2f}  v {snapshot.velocity:.3f}", (panel.left + size // 28, panel.top + size // 10), max(14, size // 25), MUTED_TEXT_COLOR)
        self._draw_text(surface, snapshot.status, (panel.left + size // 28, panel.bottom - size // 12), max(16, size // 22), _status_color(snapshot.status))
        self._draw_text(surface, action_label, (panel.left + size // 3, panel.bottom - size // 12), max(16, size // 22), ACTION_COLOR if action_value is not None else MUTED_TEXT_COLOR)

    def _draw_arrow(self, surface: Any, start: Position, vector: np.ndarray, color: RGBColor, size: int) -> None:
        import pygame

        length = float(np.linalg.norm(vector))
        if length < 1.0:
            return
        end = (int(start[0] + vector[0]), int(start[1] + vector[1]))
        pygame.draw.line(surface, color, start, end, width=max(2, size // 110))
        angle = math.atan2(vector[1], vector[0])
        arrow_size = max(7, size // 45)
        head = [
            end,
            (
                int(end[0] - arrow_size * math.cos(angle - math.pi / 6)),
                int(end[1] - arrow_size * math.sin(angle - math.pi / 6)),
            ),
            (
                int(end[0] - arrow_size * math.cos(angle + math.pi / 6)),
                int(end[1] - arrow_size * math.sin(angle + math.pi / 6)),
            ),
        ]
        pygame.draw.polygon(surface, color, head)

    def _draw_text(self, surface: Any, text: str, pos: Position, size: int, color: RGBColor) -> None:
        import pygame

        pygame.font.init()
        font = pygame.font.Font(None, int(size))
        rendered = font.render(text, True, color)
        surface.blit(rendered, pos)


class _Road1DSnapshot:
    def __init__(
        self,
        *,
        position: float,
        velocity: float,
        min_position: float,
        max_position: float,
        max_speed: float,
        speed_limit: float,
        goal_position: float,
        last_action: Any,
        step_count: int,
        status: str,
    ) -> None:
        self.position = position
        self.velocity = velocity
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.speed_limit = speed_limit
        self.goal_position = goal_position
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


def _x_to_px(position: float, snapshot: _Road1DSnapshot, track_left: int, track_right: int) -> int:
    clipped = max(snapshot.min_position, min(snapshot.max_position, position))
    ratio = (clipped - snapshot.min_position) / (snapshot.max_position - snapshot.min_position)
    return int(track_left + ratio * (track_right - track_left))


def _scalar_to_screen(value: float, max_abs: float, max_length: float) -> float:
    if max_abs <= 0:
        return 0.0
    clipped = max(-max_abs, min(max_abs, value))
    return clipped * (max_length / max_abs)


def _status(
    position: float,
    velocity: float,
    min_position: float,
    max_position: float,
    max_speed: float,
    speed_limit: float,
    goal_position: float,
) -> str:
    if position >= goal_position:
        return "goal"
    if position >= max_position or position <= min_position:
        return "boundary"
    if abs(velocity) >= max_speed:
        return "max_speed"
    if abs(velocity) >= speed_limit:
        return "over"
    return "driving"


def _status_color(status: str) -> RGBColor:
    if status == "goal":
        return GOAL_COLOR
    if status == "boundary":
        return ACTION_COLOR
    if status == "max_speed":
        return SPEED_LIMIT_COLOR
    if status == "over":
        return VELOCITY_COLOR
    return AGENT_COLOR


def _last_action_value(last_action: Any) -> float | None:
    if last_action is None:
        return None
    try:
        values = np.asarray(last_action, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.shape[0] < 1:
        return None
    return float(values[0])


__all__ = ["Road1DRenderer", "RGBColor", "validate_renderer_options"]

