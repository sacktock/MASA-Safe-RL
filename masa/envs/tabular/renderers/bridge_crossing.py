from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
RGBColor = tuple[int, int, int]

FLOOR_COLOR = (223, 218, 203)
FLOOR_ALT_COLOR = (232, 227, 213)
GRID_COLOR = (190, 182, 164)
GOAL_COLOR = (82, 162, 107)
GOAL_GLOW_COLOR = (192, 229, 190)
LAVA_COLOR = (207, 72, 54)
LAVA_DARK_COLOR = (116, 45, 46)
LAVA_GLOW_COLOR = (244, 146, 76)
START_COLOR = (235, 178, 72)
AGENT_COLOR = (60, 111, 210)
AGENT_RING_COLOR = (236, 246, 252)
AGENT_SHADOW_COLOR = (38, 72, 135)


class BridgeCrossingEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    _grid_size: int
    _state: int | None
    _start_state: int
    _goal_states: list[int]
    _lava_states: list[int]
    _step_count: int


class BridgeCrossingRenderer:
    """Renderer for the Bridge Crossing tabular gridworld family."""

    def __init__(self, env: BridgeCrossingEnv) -> None:
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
        for state in snapshot.goal_states:
            chars[_state_position(state, snapshot.grid_size)] = "G"
        for state in snapshot.lava_states:
            chars[_state_position(state, snapshot.grid_size)] = "L"
        chars[_state_position(snapshot.start_state, snapshot.grid_size)] = "S"
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

        for state in snapshot.goal_states:
            self._draw_goal_tile(surface, state, snapshot.grid_size, high_cell)
        for state in snapshot.lava_states:
            self._draw_lava_tile(surface, state, snapshot.grid_size, high_cell)

        self._draw_start_tile(surface, snapshot.start_state, snapshot.grid_size, high_cell)
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
            pygame.display.set_caption("MASA - Bridge Crossing")
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

    def _snapshot(self) -> "_BridgeCrossingSnapshot":
        grid_size = int(self.env._grid_size)
        state = int(self.env._start_state if self.env._state is None else self.env._state)
        return _BridgeCrossingSnapshot(
            grid_size=grid_size,
            agent_position=_state_position(state, grid_size),
            start_state=int(self.env._start_state),
            goal_states={int(goal) for goal in self.env._goal_states},
            lava_states={int(lava) for lava in self.env._lava_states},
            step_count=int(getattr(self.env, "_step_count", 0)),
        )

    def _draw_floor_tile(self, surface: Any, rect: Any, cell_size: int, color: RGBColor) -> None:
        import pygame

        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, GRID_COLOR, rect, width=max(1, cell_size // 36))

    def _draw_goal_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        rect = _cell_rect(state, grid_size, cell_size)
        pygame.draw.rect(surface, GOAL_COLOR, rect)
        inset = max(2, cell_size // 7)
        inner = rect.inflate(-inset * 2, -inset * 2)
        pygame.draw.rect(surface, GOAL_GLOW_COLOR, inner, border_radius=max(2, cell_size // 10))

    def _draw_lava_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        rect = _cell_rect(state, grid_size, cell_size)
        pygame.draw.rect(surface, LAVA_DARK_COLOR, rect)
        inner = rect.inflate(-max(2, cell_size // 16), -max(2, cell_size // 16))
        pygame.draw.rect(surface, LAVA_COLOR, inner, border_radius=max(2, cell_size // 12))
        center = rect.center
        radius = max(2, cell_size // 8)
        pygame.draw.circle(surface, LAVA_GLOW_COLOR, (center[0] - cell_size // 6, center[1] + cell_size // 8), radius)
        pygame.draw.circle(surface, LAVA_GLOW_COLOR, (center[0] + cell_size // 5, center[1] - cell_size // 7), max(2, radius // 2))

    def _draw_start_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        center = _state_center(state, grid_size, cell_size)
        radius = max(4, cell_size // 5)
        pygame.draw.circle(surface, (105, 76, 31), center, radius + max(2, cell_size // 24))
        pygame.draw.circle(surface, START_COLOR, center, radius)

    def _draw_agent(self, surface: Any, position: Position, cell_size: int, step_count: int) -> None:
        import pygame

        row, col = position
        center = _cell_center(row, col, cell_size)
        radius = max(5, int(cell_size * 0.32))
        bob = -max(1, cell_size // 30) if step_count % 2 else 0
        center = (center[0], center[1] + bob)
        pygame.draw.circle(surface, AGENT_SHADOW_COLOR, (center[0] + max(1, cell_size // 22), center[1] + max(1, cell_size // 22)), radius)
        pygame.draw.circle(surface, AGENT_RING_COLOR, center, radius)
        pygame.draw.circle(surface, AGENT_COLOR, center, max(2, radius - max(3, cell_size // 12)))


class _BridgeCrossingSnapshot:
    def __init__(
        self,
        *,
        grid_size: int,
        agent_position: Position,
        start_state: int,
        goal_states: set[int],
        lava_states: set[int],
        step_count: int,
    ) -> None:
        self.grid_size = grid_size
        self.agent_position = agent_position
        self.start_state = start_state
        self.goal_states = goal_states
        self.lava_states = lava_states
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


__all__ = ["BridgeCrossingRenderer", "RGBColor", "validate_renderer_options"]
