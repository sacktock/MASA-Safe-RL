from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
RGBColor = tuple[int, int, int]

FLOOR_COLOR = (236, 232, 221)
FLOOR_ALT_COLOR = (226, 222, 211)
GRID_COLOR = (194, 187, 174)
START_COLOR = (242, 178, 71)
GOAL_COLOR = (92, 172, 105)
GOAL_CENTER_COLOR = (239, 249, 232)
BLUE_COLOR = (73, 130, 210)
GREEN_COLOR = (98, 171, 99)
PURPLE_COLOR = (146, 107, 213)
AGENT_COLOR = (222, 82, 65)
AGENT_RING_COLOR = (250, 239, 226)
AGENT_SHADOW_COLOR = (143, 53, 49)


class ColourGridWorldEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    _grid_size: int
    _state: int | None
    _start_state: int
    _goal_state: int
    _blue_state: int
    _green_state: int
    _purple_state: int
    _step_count: int


class ColourGridWorldRenderer:
    """Renderer for the tabular Colour Grid World."""

    def __init__(self, env: ColourGridWorldEnv) -> None:
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
            self._human_window_size = (max(96, int(event.w)), max(96, int(event.h)))
        return True

    def _render_ansi(self) -> str:
        snapshot = self._snapshot()
        chars = np.full((snapshot.grid_size, snapshot.grid_size), " ", dtype="<U1")
        chars[_state_position(snapshot.start_state, snapshot.grid_size)] = "S"
        chars[_state_position(snapshot.goal_state, snapshot.grid_size)] = "T"
        chars[_state_position(snapshot.blue_state, snapshot.grid_size)] = "X"
        chars[_state_position(snapshot.green_state, snapshot.grid_size)] = "G"
        chars[_state_position(snapshot.purple_state, snapshot.grid_size)] = "P"
        chars[snapshot.agent_position] = "A"
        return "\n".join("".join(row) for row in chars)

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        snapshot = self._snapshot()
        cell_size = max(12, int(self.env.render_window_size) // snapshot.grid_size)
        scale = 3
        high_cell = cell_size * scale
        high_size = (snapshot.grid_size * high_cell, snapshot.grid_size * high_cell)
        surface = pygame.Surface(high_size)
        surface.fill(FLOOR_COLOR)

        for row in range(snapshot.grid_size):
            for col in range(snapshot.grid_size):
                rect = pygame.Rect(col * high_cell, row * high_cell, high_cell, high_cell)
                floor = FLOOR_COLOR if (row + col) % 2 == 0 else FLOOR_ALT_COLOR
                self._draw_floor_tile(surface, rect, high_cell, floor)

        self._draw_start_tile(surface, snapshot.start_state, snapshot.grid_size, high_cell)
        self._draw_goal_tile(surface, snapshot.goal_state, snapshot.grid_size, high_cell)
        self._draw_special_tile(surface, snapshot.blue_state, snapshot.grid_size, high_cell, BLUE_COLOR, "diamond")
        self._draw_special_tile(surface, snapshot.green_state, snapshot.grid_size, high_cell, GREEN_COLOR, "circle")
        self._draw_special_tile(surface, snapshot.purple_state, snapshot.grid_size, high_cell, PURPLE_COLOR, "square")
        self._draw_agent(surface, snapshot.agent_position, high_cell, snapshot.step_count)

        if scale > 1:
            final_size = (snapshot.grid_size * cell_size, snapshot.grid_size * cell_size)
            surface = pygame.transform.smoothscale(surface, final_size)

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
            pygame.display.set_caption("MASA - Colour Grid World")
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

        self._human_window.fill(FLOOR_COLOR)
        self._human_window.blit(frame_surface, (left, top))
        pygame.display.flip()
        if self._human_clock is not None:
            self._human_clock.tick(self.env.metadata["render_fps"])

    def _snapshot(self) -> "_ColourGridSnapshot":
        state = int(self.env._start_state if self.env._state is None else self.env._state)
        return _ColourGridSnapshot(
            grid_size=int(self.env._grid_size),
            agent_position=_state_position(state, int(self.env._grid_size)),
            start_state=int(self.env._start_state),
            goal_state=int(self.env._goal_state),
            blue_state=int(self.env._blue_state),
            green_state=int(self.env._green_state),
            purple_state=int(self.env._purple_state),
            step_count=int(getattr(self.env, "_step_count", 0)),
        )

    def _draw_floor_tile(self, surface: Any, rect: Any, cell_size: int, color: RGBColor) -> None:
        import pygame

        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, GRID_COLOR, rect, width=max(1, cell_size // 36))

    def _draw_start_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        center = _state_center(state, grid_size, cell_size)
        radius = max(5, cell_size // 5)
        pygame.draw.circle(surface, (105, 78, 31), center, radius + max(2, cell_size // 24))
        pygame.draw.circle(surface, START_COLOR, center, radius)

    def _draw_goal_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        rect = _cell_rect(state, grid_size, cell_size).inflate(-cell_size // 6, -cell_size // 6)
        pygame.draw.rect(surface, GOAL_COLOR, rect, border_radius=max(4, cell_size // 10))
        inner = rect.inflate(-cell_size // 4, -cell_size // 4)
        pygame.draw.rect(surface, GOAL_CENTER_COLOR, inner, border_radius=max(2, cell_size // 16))

    def _draw_special_tile(self, surface: Any, state: int, grid_size: int, cell_size: int, color: RGBColor, shape: str) -> None:
        import pygame

        rect = _cell_rect(state, grid_size, cell_size)
        center = rect.center
        radius = max(6, cell_size // 3)
        shadow = _mix(color, (0, 0, 0), 0.35)
        if shape == "diamond":
            points = [
                (center[0], center[1] - radius),
                (center[0] + radius, center[1]),
                (center[0], center[1] + radius),
                (center[0] - radius, center[1]),
            ]
            pygame.draw.polygon(surface, shadow, [(x + 2, y + 2) for x, y in points])
            pygame.draw.polygon(surface, color, points)
            return
        if shape == "circle":
            pygame.draw.circle(surface, shadow, (center[0] + 2, center[1] + 2), radius)
            pygame.draw.circle(surface, color, center, radius)
            return

        body = rect.inflate(-cell_size // 4, -cell_size // 4)
        pygame.draw.rect(surface, shadow, body.move(2, 2), border_radius=max(4, cell_size // 10))
        pygame.draw.rect(surface, color, body, border_radius=max(4, cell_size // 10))

    def _draw_agent(self, surface: Any, position: Position, cell_size: int, step_count: int) -> None:
        import pygame

        row, col = position
        center = _cell_center(row, col, cell_size)
        radius = max(6, int(cell_size * 0.32))
        bob = -max(1, cell_size // 30) if step_count % 2 else 0
        center = (center[0], center[1] + bob)
        pygame.draw.circle(surface, AGENT_SHADOW_COLOR, (center[0] + max(1, cell_size // 22), center[1] + max(1, cell_size // 22)), radius)
        pygame.draw.circle(surface, AGENT_RING_COLOR, center, radius)
        pygame.draw.circle(surface, AGENT_COLOR, center, max(2, radius - max(3, cell_size // 12)))


class _ColourGridSnapshot:
    def __init__(
        self,
        *,
        grid_size: int,
        agent_position: Position,
        start_state: int,
        goal_state: int,
        blue_state: int,
        green_state: int,
        purple_state: int,
        step_count: int,
    ) -> None:
        self.grid_size = grid_size
        self.agent_position = agent_position
        self.start_state = start_state
        self.goal_state = goal_state
        self.blue_state = blue_state
        self.green_state = green_state
        self.purple_state = purple_state
        self.step_count = step_count


def validate_renderer_options(render_mode: str | None, render_window_size: int) -> None:
    if render_mode not in (None, "ansi", "rgb_array", "human"):
        raise ValueError("render_mode must be None, 'ansi', 'rgb_array', or 'human'.")
    if int(render_window_size) <= 0:
        raise ValueError("render_window_size must be positive.")


def _state_position(state: int, grid_size: int) -> Position:
    return divmod(int(state), grid_size)


def _cell_center(row: int, col: int, cell_size: int) -> Position:
    return col * cell_size + cell_size // 2, row * cell_size + cell_size // 2


def _state_center(state: int, grid_size: int, cell_size: int) -> Position:
    row, col = _state_position(state, grid_size)
    return _cell_center(row, col, cell_size)


def _cell_rect(state: int, grid_size: int, cell_size: int) -> Any:
    import pygame

    row, col = _state_position(state, grid_size)
    return pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)


def _fit_size(source: Position, target: Position) -> Position:
    source_width, source_height = source
    target_width, target_height = target
    scale = min(target_width / source_width, target_height / source_height)
    return max(1, int(source_width * scale)), max(1, int(source_height * scale))


def _mix(color: RGBColor, other: RGBColor, ratio: float) -> RGBColor:
    return tuple(int(channel * (1.0 - ratio) + other_channel * ratio) for channel, other_channel in zip(color, other))


__all__ = ["ColourGridWorldRenderer", "RGBColor", "validate_renderer_options"]
