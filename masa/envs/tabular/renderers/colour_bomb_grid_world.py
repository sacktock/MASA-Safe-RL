from __future__ import annotations

import os
from typing import Any, Literal, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
RGBColor = tuple[int, int, int]
RenderMode = Literal["ansi", "rgb_array", "human"]

FLOOR_COLOR = (235, 232, 221)
FLOOR_ALT_COLOR = (226, 222, 211)
GRID_COLOR = (193, 187, 174)
WALL_COLOR = (68, 78, 92)
WALL_EDGE_COLOR = (43, 50, 62)
START_COLOR = (245, 188, 82)
BOMB_COLOR = (34, 39, 48)
BOMB_CORE_COLOR = (11, 15, 22)
BOMB_HIGHLIGHT_COLOR = (111, 123, 140)
BOMB_CAP_COLOR = (150, 155, 164)
BOMB_FUSE_COLOR = (73, 56, 38)
BOMB_SPARK_COLOR = (255, 191, 54)
BOMB_SPARK_CORE_COLOR = (255, 246, 167)
MEDIC_COLOR = (78, 166, 129)
MEDIC_CROSS_COLOR = (238, 248, 236)
AGENT_COLOR = (55, 115, 206)
AGENT_RING_COLOR = (236, 246, 252)
AGENT_SHADOW_COLOR = (38, 75, 136)
ACTIVE_OUTLINE_COLOR = (33, 30, 38)
ZONE_COLORS: dict[str, RGBColor] = {
    "green": (109, 178, 103),
    "yellow": (230, 200, 81),
    "red": (216, 93, 83),
    "blue": (89, 143, 219),
    "pink": (213, 116, 176),
}
ZONE_MARKERS = {
    "green": "G",
    "yellow": "Y",
    "red": "R",
    "blue": "B",
    "pink": "P",
}


class ColourBombGridWorldEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    _grid_size: int
    _state: int | None
    _step_count: int
    _start_state: int
    _start_states: list[int]
    _green_states: list[int]
    _yellow_states: list[int]
    _red_states: list[int]
    _blue_states: list[int]
    _pink_states: list[int]
    _wall_states: list[int]
    _bomb_states: list[int]
    _medic_states: list[int]
    _active_colour_dict: dict[int, str]


class ColourBombGridWorldRenderer:
    """Renderer for the Colour Bomb tabular gridworld family."""

    def __init__(self, env: ColourBombGridWorldEnv) -> None:
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

        for state in snapshot.wall_cells:
            chars[_state_position(state, snapshot.grid_size)] = "#"
        for colour, marker in ZONE_MARKERS.items():
            for state in snapshot.colour_cells[colour]:
                chars[_state_position(state, snapshot.grid_size)] = marker
        for state in snapshot.start_cells:
            chars[_state_position(state, snapshot.grid_size)] = "S"
        for state in snapshot.medic_cells:
            chars[_state_position(state, snapshot.grid_size)] = "M"
        for state in snapshot.bomb_cells:
            chars[_state_position(state, snapshot.grid_size)] = "X"

        chars[snapshot.agent_position] = "A"
        return "\n".join("".join(row) for row in chars)

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        snapshot = self._snapshot()
        cell_size = max(12, int(self.env.render_window_size) // snapshot.grid_size)
        scale = 3
        high_cell = cell_size * scale
        surface_size = (snapshot.grid_size * high_cell, snapshot.grid_size * high_cell)
        surface = pygame.Surface(surface_size)
        surface.fill(FLOOR_COLOR)

        for row in range(snapshot.grid_size):
            for col in range(snapshot.grid_size):
                rect = pygame.Rect(col * high_cell, row * high_cell, high_cell, high_cell)
                floor = FLOOR_COLOR if (row + col) % 2 == 0 else FLOOR_ALT_COLOR
                self._draw_floor_tile(surface, rect, high_cell, floor)

        for colour, cells in snapshot.colour_cells.items():
            for state in cells:
                self._draw_colour_tile(surface, state, snapshot.grid_size, high_cell, ZONE_COLORS[colour])

        if snapshot.active_colour is not None:
            for state in snapshot.colour_cells[snapshot.active_colour]:
                self._draw_active_outline(surface, state, snapshot.grid_size, high_cell)

        for state in snapshot.start_cells:
            self._draw_start_tile(surface, state, snapshot.grid_size, high_cell)
        for state in snapshot.medic_cells:
            self._draw_medic_tile(surface, state, snapshot.grid_size, high_cell)
        for state in snapshot.bomb_cells:
            self._draw_bomb_tile(surface, state, snapshot.grid_size, high_cell)
        for state in snapshot.wall_cells:
            self._draw_wall_tile(surface, state, snapshot.grid_size, high_cell)

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
            pygame.display.set_caption("MASA - Colour Bomb Grid World")
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

    def _snapshot(self) -> "_ColourBombSnapshot":
        grid_size = int(self.env._grid_size)
        grid_area = grid_size**2
        state = int(self.env._start_state if self.env._state is None else self.env._state)
        zone = state // grid_area
        active_colour = getattr(self.env, "_active_colour_dict", {}).get(zone)

        return _ColourBombSnapshot(
            grid_size=grid_size,
            agent_position=_state_position(state % grid_area, grid_size),
            wall_cells=_base_cells(getattr(self.env, "_wall_states", []), grid_area),
            start_cells=_base_cells(_start_states(self.env), grid_area),
            bomb_cells=_base_cells(getattr(self.env, "_bomb_states", []), grid_area),
            medic_cells=_base_cells(getattr(self.env, "_medic_states", []), grid_area),
            colour_cells={
                "green": _base_cells(getattr(self.env, "_green_states", []), grid_area),
                "yellow": _base_cells(getattr(self.env, "_yellow_states", []), grid_area),
                "red": _base_cells(getattr(self.env, "_red_states", []), grid_area),
                "blue": _base_cells(getattr(self.env, "_blue_states", []), grid_area),
                "pink": _base_cells(getattr(self.env, "_pink_states", []), grid_area),
            },
            active_colour=active_colour,
            step_count=int(getattr(self.env, "_step_count", 0)),
        )

    def _draw_floor_tile(self, surface: Any, rect: Any, cell_size: int, color: RGBColor) -> None:
        import pygame

        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, GRID_COLOR, rect, width=max(1, cell_size // 36))

    def _draw_wall_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        rect = _cell_rect(state, grid_size, cell_size)
        pygame.draw.rect(surface, WALL_EDGE_COLOR, rect, border_radius=max(3, cell_size // 12))
        body = rect.inflate(-max(2, cell_size // 18), -max(2, cell_size // 18))
        pygame.draw.rect(surface, WALL_COLOR, body, border_radius=max(3, cell_size // 12))

    def _draw_colour_tile(self, surface: Any, state: int, grid_size: int, cell_size: int, color: RGBColor) -> None:
        import pygame

        rect = _cell_rect(state, grid_size, cell_size)
        body = rect.inflate(-max(3, cell_size // 10), -max(3, cell_size // 10))
        pygame.draw.rect(surface, color, body, border_radius=max(3, cell_size // 10))
        gloss = body.inflate(-body.width // 3, -body.height // 3)
        highlight = _mix(color, (255, 255, 255), 0.35)
        pygame.draw.rect(surface, highlight, gloss, border_radius=max(2, cell_size // 14))

    def _draw_active_outline(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        rect = _cell_rect(state, grid_size, cell_size).inflate(-max(2, cell_size // 18), -max(2, cell_size // 18))
        pygame.draw.rect(surface, ACTIVE_OUTLINE_COLOR, rect, width=max(3, cell_size // 16), border_radius=max(4, cell_size // 9))

    def _draw_start_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        center = _state_center(state, grid_size, cell_size)
        radius = max(4, cell_size // 5)
        pygame.draw.circle(surface, (101, 76, 31), center, radius + max(2, cell_size // 24))
        pygame.draw.circle(surface, START_COLOR, center, radius)

    def _draw_bomb_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        center = _state_center(state, grid_size, cell_size)
        radius = max(5, int(cell_size * 0.28))
        shadow_offset = max(1, cell_size // 22)
        outline = max(1, cell_size // 28)
        pygame.draw.circle(surface, BOMB_CORE_COLOR, (center[0] + shadow_offset, center[1] + shadow_offset), radius)
        pygame.draw.circle(surface, BOMB_COLOR, center, radius)
        pygame.draw.circle(surface, BOMB_CORE_COLOR, center, radius, width=outline)

        highlight_radius = max(2, radius // 4)
        highlight = (center[0] - radius // 3, center[1] - radius // 3)
        pygame.draw.circle(surface, BOMB_HIGHLIGHT_COLOR, highlight, highlight_radius)

        cap_width = max(5, cell_size // 5)
        cap_height = max(3, cell_size // 9)
        cap_left = center[0] + radius // 4
        cap_top = center[1] - radius - cap_height // 2
        cap = pygame.Rect(cap_left, cap_top, cap_width, cap_height)
        pygame.draw.rect(surface, BOMB_CORE_COLOR, cap.inflate(outline * 2, outline * 2), border_radius=max(2, cell_size // 24))
        pygame.draw.rect(surface, BOMB_CAP_COLOR, cap, border_radius=max(2, cell_size // 24))

        fuse_start = (cap.centerx + cap_width // 3, cap.top)
        fuse_mid = (fuse_start[0] + max(3, cell_size // 8), fuse_start[1] - max(4, cell_size // 7))
        fuse_end = (fuse_mid[0] + max(3, cell_size // 9), fuse_mid[1] + max(1, cell_size // 18))
        pygame.draw.lines(surface, BOMB_FUSE_COLOR, False, (fuse_start, fuse_mid, fuse_end), width=max(2, cell_size // 24))

        spark_radius = max(3, cell_size // 14)
        for dx, dy in ((spark_radius, 0), (-spark_radius, 0), (0, spark_radius), (0, -spark_radius)):
            pygame.draw.line(surface, BOMB_SPARK_COLOR, fuse_end, (fuse_end[0] + dx, fuse_end[1] + dy), width=max(1, cell_size // 36))
        pygame.draw.circle(surface, BOMB_SPARK_COLOR, fuse_end, max(2, spark_radius // 2))
        pygame.draw.circle(surface, BOMB_SPARK_CORE_COLOR, fuse_end, max(1, spark_radius // 3))

    def _draw_medic_tile(self, surface: Any, state: int, grid_size: int, cell_size: int) -> None:
        import pygame

        rect = _cell_rect(state, grid_size, cell_size).inflate(-cell_size // 5, -cell_size // 5)
        pygame.draw.rect(surface, MEDIC_COLOR, rect, border_radius=max(3, cell_size // 12))
        cx, cy = rect.center
        bar = max(3, cell_size // 9)
        length = max(8, cell_size // 2)
        pygame.draw.rect(surface, MEDIC_CROSS_COLOR, pygame.Rect(cx - bar // 2, cy - length // 2, bar, length))
        pygame.draw.rect(surface, MEDIC_CROSS_COLOR, pygame.Rect(cx - length // 2, cy - bar // 2, length, bar))

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


class _ColourBombSnapshot:
    def __init__(
        self,
        *,
        grid_size: int,
        agent_position: Position,
        wall_cells: set[int],
        start_cells: set[int],
        bomb_cells: set[int],
        medic_cells: set[int],
        colour_cells: dict[str, set[int]],
        active_colour: str | None,
        step_count: int,
    ) -> None:
        self.grid_size = grid_size
        self.agent_position = agent_position
        self.wall_cells = wall_cells
        self.start_cells = start_cells
        self.bomb_cells = bomb_cells
        self.medic_cells = medic_cells
        self.colour_cells = colour_cells
        self.active_colour = active_colour
        self.step_count = step_count


def validate_renderer_options(render_mode: str | None, render_window_size: int) -> None:
    if render_mode not in (None, "ansi", "rgb_array", "human"):
        raise ValueError("render_mode must be None, 'ansi', 'rgb_array', or 'human'.")
    if int(render_window_size) <= 0:
        raise ValueError("render_window_size must be positive.")


def _start_states(env: ColourBombGridWorldEnv) -> list[int]:
    if hasattr(env, "_start_states"):
        return list(env._start_states)
    return [int(env._start_state)]


def _base_cells(states: list[int], grid_area: int) -> set[int]:
    return {int(state) % grid_area for state in states}


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


__all__ = ["ColourBombGridWorldRenderer", "RenderMode", "RGBColor", "validate_renderer_options"]
