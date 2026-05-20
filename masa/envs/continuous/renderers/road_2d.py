from __future__ import annotations

import math
import os
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
RGBColor = tuple[int, int, int]

BACKGROUND_COLOR = (232, 235, 229)
PANEL_COLOR = (217, 224, 217)
ROAD_COLOR = (72, 81, 86)
ROAD_EDGE_COLOR = (241, 244, 236)
ROAD_SHOULDER_COLOR = (137, 149, 145)
LANE_MARKING_COLOR = (246, 226, 131)
GRID_COLOR = (122, 135, 133)
GOAL_COLOR = (92, 166, 111)
GOAL_FILL_COLOR = (118, 184, 128)
BOUNDARY_COLOR = (38, 47, 58)
AGENT_COLOR = (66, 121, 210)
AGENT_SHADOW_COLOR = (39, 67, 126)
VELOCITY_COLOR = (198, 126, 63)
ACTION_COLOR = (202, 76, 75)
SPEED_LIMIT_COLOR = (235, 187, 86)
TEXT_COLOR = (39, 45, 58)
MUTED_TEXT_COLOR = (95, 103, 111)


class Road2DEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    action_space: Any
    _state: np.ndarray
    _min_position: float
    _max_position: float
    _max_speed: float
    _goal_position: np.ndarray
    _speed_limit: float
    _last_action: Any
    _step_count: int


class Road2DRenderer:
    """Renderer for the continuous Road2D environment."""

    def __init__(self, env: Road2DEnv) -> None:
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
            f"position=({snapshot.position[0]:.3f}, {snapshot.position[1]:.3f})\n"
            f"velocity=({snapshot.velocity[0]:.3f}, {snapshot.velocity[1]:.3f})\n"
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
            pygame.display.set_caption("MASA - Road2D")
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

    def _snapshot(self) -> "_Road2DSnapshot":
        state = np.zeros(4, dtype=np.float32)
        if getattr(self.env, "_state", None) is not None:
            state = np.asarray(self.env._state, dtype=np.float32)
        min_position = float(self.env._min_position)
        max_position = float(self.env._max_position)
        max_speed = float(self.env._max_speed)
        speed_limit = float(self.env._speed_limit)
        goal_position = np.asarray(self.env._goal_position, dtype=np.float32)
        position = np.asarray(state[:2], dtype=np.float32)
        velocity = np.asarray(state[2:], dtype=np.float32)
        status = _status(position, velocity, min_position, max_position, max_speed, speed_limit, goal_position)
        return _Road2DSnapshot(
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

    def _draw_scene(self, surface: Any, snapshot: "_Road2DSnapshot", size: int) -> None:
        import pygame

        margin = max(24, size // 14)
        panel = pygame.Rect(margin, margin, size - margin * 2, size - margin * 2)
        pygame.draw.rect(surface, PANEL_COLOR, panel, border_radius=max(8, size // 32))
        pygame.draw.rect(surface, ROAD_EDGE_COLOR, panel, width=max(2, size // 180), border_radius=max(8, size // 32))

        map_margin = max(22, size // 13)
        map_rect = pygame.Rect(
            panel.left + map_margin,
            panel.top + map_margin + size // 18,
            panel.width - map_margin * 2,
            panel.height - map_margin * 2 - size // 12,
        )
        shoulder_rect = map_rect.inflate(max(16, size // 34), max(16, size // 34))
        pygame.draw.rect(surface, ROAD_SHOULDER_COLOR, shoulder_rect, border_radius=max(10, size // 48))
        pygame.draw.rect(surface, ROAD_EDGE_COLOR, shoulder_rect, width=max(2, size // 120), border_radius=max(10, size // 48))
        pygame.draw.rect(surface, ROAD_COLOR, map_rect, border_radius=max(6, size // 70))
        pygame.draw.rect(surface, BOUNDARY_COLOR, map_rect, width=max(3, size // 110), border_radius=max(6, size // 70))

        self._draw_grid(surface, snapshot, map_rect, size)
        self._draw_goal_region(surface, snapshot, map_rect)

        agent = _point_for_position(snapshot.position, snapshot, map_rect)
        velocity_vector = _vector_to_screen(snapshot.velocity, snapshot.max_speed, map_rect.width * 0.18)
        self._draw_arrow(surface, agent, velocity_vector, VELOCITY_COLOR, size)

        action_vector = _last_action_vector(snapshot.last_action)
        if action_vector is not None:
            self._draw_arrow(surface, agent, _vector_to_screen(action_vector, 2.0, map_rect.width * 0.12), ACTION_COLOR, size)

        agent_radius = max(8, size // 40)
        pygame.draw.circle(surface, AGENT_SHADOW_COLOR, (agent[0] + max(2, size // 120), agent[1] + max(2, size // 120)), agent_radius)
        pygame.draw.circle(surface, _status_color(snapshot.status), agent, agent_radius)
        pygame.draw.circle(surface, TEXT_COLOR, agent, agent_radius, width=max(2, size // 150))

        action_label = "last action: none" if action_vector is None else f"last action: ({action_vector[0]:.2f}, {action_vector[1]:.2f})"
        self._draw_text(surface, "Road2D", (panel.left + size // 28, panel.top + size // 28), max(18, size // 18), TEXT_COLOR)
        self._draw_text(surface, f"x {snapshot.position[0]:.2f}, {snapshot.position[1]:.2f}  v {np.linalg.norm(snapshot.velocity):.3f}", (panel.left + size // 28, panel.top + size // 10), max(14, size // 25), MUTED_TEXT_COLOR)
        self._draw_text(surface, snapshot.status, (panel.left + size // 28, panel.bottom - size // 12), max(16, size // 22), _status_color(snapshot.status))
        self._draw_text(surface, action_label, (panel.left + size // 3, panel.bottom - size // 12), max(16, size // 22), ACTION_COLOR if action_vector is not None else MUTED_TEXT_COLOR)

    def _draw_grid(self, surface: Any, snapshot: "_Road2DSnapshot", map_rect: Any, size: int) -> None:
        import pygame

        edge_inset = max(5, size // 72)
        pygame.draw.line(
            surface,
            ROAD_EDGE_COLOR,
            (map_rect.left + edge_inset, map_rect.top + edge_inset),
            (map_rect.right - edge_inset, map_rect.top + edge_inset),
            width=max(2, size // 150),
        )
        pygame.draw.line(
            surface,
            ROAD_EDGE_COLOR,
            (map_rect.left + edge_inset, map_rect.bottom - edge_inset),
            (map_rect.right - edge_inset, map_rect.bottom - edge_inset),
            width=max(2, size // 150),
        )
        pygame.draw.line(
            surface,
            ROAD_EDGE_COLOR,
            (map_rect.left + edge_inset, map_rect.top + edge_inset),
            (map_rect.left + edge_inset, map_rect.bottom - edge_inset),
            width=max(2, size // 150),
        )
        pygame.draw.line(
            surface,
            ROAD_EDGE_COLOR,
            (map_rect.right - edge_inset, map_rect.top + edge_inset),
            (map_rect.right - edge_inset, map_rect.bottom - edge_inset),
            width=max(2, size // 150),
        )

        for value in range(math.ceil(snapshot.min_position), math.floor(snapshot.max_position) + 1):
            x = _x_to_px(float(value), snapshot, map_rect)
            y = _y_to_px(float(value), snapshot, map_rect)
            if value == 0:
                _draw_dashed_line(
                    surface,
                    (x, map_rect.top + edge_inset),
                    (x, map_rect.bottom - edge_inset),
                    LANE_MARKING_COLOR,
                    width=max(2, size // 130),
                    dash_length=max(10, size // 34),
                    gap_length=max(8, size // 48),
                )
                _draw_dashed_line(
                    surface,
                    (map_rect.left + edge_inset, y),
                    (map_rect.right - edge_inset, y),
                    LANE_MARKING_COLOR,
                    width=max(2, size // 130),
                    dash_length=max(10, size // 34),
                    gap_length=max(8, size // 48),
                )
                continue
            pygame.draw.line(surface, GRID_COLOR, (x, map_rect.top + edge_inset), (x, map_rect.bottom - edge_inset), width=max(1, size // 260))
            pygame.draw.line(surface, GRID_COLOR, (map_rect.left + edge_inset, y), (map_rect.right - edge_inset, y), width=max(1, size // 260))

    def _draw_goal_region(self, surface: Any, snapshot: "_Road2DSnapshot", map_rect: Any) -> None:
        import pygame

        goal_left = _x_to_px(float(snapshot.goal_position[0]), snapshot, map_rect)
        goal_bottom = _y_to_px(float(snapshot.goal_position[1]), snapshot, map_rect)
        goal_rect = pygame.Rect(goal_left, map_rect.top, map_rect.right - goal_left, goal_bottom - map_rect.top)
        _draw_translucent_rect(surface, goal_rect, GOAL_FILL_COLOR, alpha=120)
        pygame.draw.rect(surface, GOAL_COLOR, goal_rect, width=max(3, map_rect.width // 100))

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


class _Road2DSnapshot:
    def __init__(
        self,
        *,
        position: np.ndarray,
        velocity: np.ndarray,
        min_position: float,
        max_position: float,
        max_speed: float,
        speed_limit: float,
        goal_position: np.ndarray,
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


def _draw_translucent_rect(surface: Any, rect: Any, color: RGBColor, *, alpha: int, border_radius: int = 0) -> None:
    import pygame

    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(overlay, (*color, alpha), overlay.get_rect(), border_radius=border_radius)
    surface.blit(overlay, rect.topleft)


def _draw_dashed_line(
    surface: Any,
    start: Position,
    end: Position,
    color: RGBColor,
    *,
    width: int,
    dash_length: int,
    gap_length: int,
) -> None:
    import pygame

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        return

    step_x = dx / length
    step_y = dy / length
    cursor = 0.0
    while cursor < length:
        next_cursor = min(cursor + dash_length, length)
        segment_start = (int(start[0] + step_x * cursor), int(start[1] + step_y * cursor))
        segment_end = (int(start[0] + step_x * next_cursor), int(start[1] + step_y * next_cursor))
        pygame.draw.line(surface, color, segment_start, segment_end, width=width)
        cursor += dash_length + gap_length


def _point_for_position(position: np.ndarray, snapshot: _Road2DSnapshot, map_rect: Any) -> Position:
    return _x_to_px(float(position[0]), snapshot, map_rect), _y_to_px(float(position[1]), snapshot, map_rect)


def _x_to_px(x: float, snapshot: _Road2DSnapshot, map_rect: Any) -> int:
    clipped = max(snapshot.min_position, min(snapshot.max_position, x))
    ratio = (clipped - snapshot.min_position) / (snapshot.max_position - snapshot.min_position)
    return int(map_rect.left + ratio * map_rect.width)


def _y_to_px(y: float, snapshot: _Road2DSnapshot, map_rect: Any) -> int:
    clipped = max(snapshot.min_position, min(snapshot.max_position, y))
    ratio = (clipped - snapshot.min_position) / (snapshot.max_position - snapshot.min_position)
    return int(map_rect.bottom - ratio * map_rect.height)


def _vector_to_screen(vector: np.ndarray, max_norm: float, max_length: float) -> np.ndarray:
    if max_norm <= 0:
        return np.zeros(2, dtype=np.float32)
    clipped = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(clipped))
    if norm > max_norm:
        clipped = clipped * (max_norm / norm)
    return np.array([clipped[0], -clipped[1]], dtype=np.float32) * (max_length / max_norm)


def _status(
    position: np.ndarray,
    velocity: np.ndarray,
    min_position: float,
    max_position: float,
    max_speed: float,
    speed_limit: float,
    goal_position: np.ndarray,
) -> str:
    if np.all(position >= goal_position):
        return "goal"
    if np.any(position >= max_position) or np.any(position <= min_position):
        return "boundary"
    if np.any(np.abs(velocity) >= max_speed):
        return "max_speed"
    if np.any(np.abs(velocity) >= speed_limit):
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


def _last_action_vector(last_action: Any) -> np.ndarray | None:
    if last_action is None:
        return None
    try:
        values = np.asarray(last_action, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.shape[0] < 2:
        return None
    return values[:2]


__all__ = ["Road2DRenderer", "RGBColor", "validate_renderer_options"]
