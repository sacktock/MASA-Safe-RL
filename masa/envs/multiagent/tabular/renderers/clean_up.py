from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
Color = tuple[int, int, int]

SAND: Color = (220, 218, 186)
SAND_DARK: Color = (205, 201, 171)
GRASS: Color = (118, 158, 65)
GRASS_LIGHT: Color = (164, 189, 75)
WATER: Color = (35, 133, 168)
WATER_LIGHT: Color = (66, 173, 212)
WALL: Color = (92, 94, 98)
DIRT: Color = (60, 126, 72)
APPLE: Color = (212, 80, 57)
ZAP: Color = (252, 252, 106)
CLEAN: Color = (99, 223, 242)
WHITE: Color = (245, 245, 235)
BLACK: Color = (0, 0, 0)
PLAYER_COLORS: tuple[Color, ...] = (
    (45, 110, 220),
    (125, 50, 200),
    (205, 5, 165),
    (245, 65, 65),
    (245, 130, 0),
    (195, 180, 0),
    (125, 185, 65),
)


class CleanUpEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    _rows: int
    _cols: int
    _grid: list[str]
    _walls: set[Position]
    _grass: set[Position]
    _grass_edges: set[Position]
    _river: set[Position]
    _apples: dict[Position, Any]
    _dirts: dict[Position, Any]
    _avatars: dict[str, Any]
    _last_zap_cells: set[Position]
    _last_clean_cells: set[Position]


class CleanUpRenderer:
    def __init__(self, env: CleanUpEnv) -> None:
        self.env = env
        self._window = None
        self._clock = None
        self._window_closed = False

    @property
    def human_window_closed(self) -> bool:
        return self._window_closed

    def render(self) -> str | np.ndarray | None:
        if self.env.render_mode is None:
            return None
        if self.env.render_mode == "ansi":
            return self._render_ansi()
        frame = self._render_rgb_array()
        if self.env.render_mode == "rgb_array":
            return frame
        self._render_human(frame)
        return None

    def close(self) -> None:
        if self._window is not None:
            import pygame

            pygame.display.quit()
        self._window = None
        self._clock = None
        self._window_closed = True

    def handle_pygame_event(self, event: Any) -> bool:
        import pygame

        if event.type == pygame.QUIT:
            self.close()
            return False
        return True

    def _render_ansi(self) -> str:
        chars = [list(row) for row in self.env._grid]
        for dirt in self.env._dirts.values():
            if dirt.active:
                x, y = dirt.position
                chars[y][x] = "X"
        for apple in self.env._apples.values():
            if apple.live:
                x, y = apple.position
                chars[y][x] = "A"
        for avatar in self.env._avatars.values():
            if avatar.active:
                x, y = avatar.position
                chars[y][x] = str(avatar.index % 10)
        return "\n".join("".join(row) for row in chars)

    def _render_rgb_array(self) -> np.ndarray:
        cell = max(4, self.env.render_window_size // max(self.env._rows, self.env._cols))
        frame = np.empty((self.env._rows * cell, self.env._cols * cell, 3), dtype=np.uint8)
        frame[:] = BLACK
        for y in range(self.env._rows):
            for x in range(self.env._cols):
                position = (x, y)
                if position in self.env._walls:
                    color = WALL
                elif position in self.env._river:
                    color = WATER if y % 2 else WATER_LIGHT
                elif position in self.env._grass:
                    color = GRASS if (x + y) % 2 else GRASS_LIGHT
                elif position in self.env._grass_edges:
                    color = GRASS_LIGHT
                else:
                    color = SAND if (x + y) % 2 else SAND_DARK
                self._fill(frame, position, cell, color)

        for apple in self.env._apples.values():
            if apple.live:
                self._draw_apple(frame, apple.position, cell)
        for dirt in self.env._dirts.values():
            if dirt.active:
                self._draw_dirt(frame, dirt.position, cell)
        for position in self.env._last_zap_cells:
            self._draw_beam(frame, position, cell, ZAP)
        for position in self.env._last_clean_cells:
            self._draw_beam(frame, position, cell, CLEAN)
        for avatar in self.env._avatars.values():
            if avatar.active:
                self._draw_avatar(frame, avatar, cell)
        return frame

    def _render_human(self, frame: np.ndarray) -> None:
        if self._window_closed:
            return
        import pygame

        if self._window is None:
            pygame.init()
            pygame.display.set_caption("MASA - Clean Up")
            height, width = frame.shape[:2]
            self._window = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            self._clock = pygame.time.Clock()
        for event in pygame.event.get():
            self.handle_pygame_event(event)
        if self._window is None:
            return
        width, height = self._window.get_size()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        if surface.get_size() != (width, height):
            surface = pygame.transform.smoothscale(surface, (width, height))
        self._window.blit(surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.env.metadata["render_fps"])

    @staticmethod
    def _fill(frame: np.ndarray, position: Position, cell: int, color: Color) -> None:
        x, y = position
        frame[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = color

    def _draw_apple(self, frame: np.ndarray, position: Position, cell: int) -> None:
        x, y = position
        y0, x0 = y * cell, x * cell
        pad = max(2, cell // 4)
        frame[y0 + pad : y0 + cell - pad, x0 + pad : x0 + cell - pad] = APPLE

    def _draw_dirt(self, frame: np.ndarray, position: Position, cell: int) -> None:
        x, y = position
        y0, x0 = y * cell, x * cell
        pad = max(1, cell // 7)
        frame[y0 + pad : y0 + cell - pad, x0 + pad : x0 + cell - pad] = DIRT

    def _draw_beam(
        self,
        frame: np.ndarray,
        position: Position,
        cell: int,
        color: Color,
    ) -> None:
        x, y = position
        y0, x0 = y * cell, x * cell
        mid = cell // 2
        frame[y0 : y0 + cell, x0 + mid : x0 + mid + 1] = color
        frame[y0 + mid : y0 + mid + 1, x0 : x0 + cell] = color

    def _draw_avatar(self, frame: np.ndarray, avatar: Any, cell: int) -> None:
        x, y = avatar.position
        y0, x0 = y * cell, x * cell
        pad = max(2, cell // 6)
        color = PLAYER_COLORS[avatar.index % len(PLAYER_COLORS)]
        frame[y0 + pad : y0 + cell - pad, x0 + pad : x0 + cell - pad] = color
        mid = cell // 2
        tip = max(2, cell // 4)
        if avatar.orientation == 0:
            frame[y0 + pad : y0 + pad + tip, x0 + mid : x0 + mid + 2] = WHITE
        elif avatar.orientation == 1:
            frame[y0 + mid : y0 + mid + 2, x0 + cell - pad - tip : x0 + cell - pad] = WHITE
        elif avatar.orientation == 2:
            frame[y0 + cell - pad - tip : y0 + cell - pad, x0 + mid : x0 + mid + 2] = WHITE
        else:
            frame[y0 + mid : y0 + mid + 2, x0 + pad : x0 + pad + tip] = WHITE


def validate_renderer_options(render_mode: str | None, render_window_size: int) -> None:
    if render_mode not in (None, "ansi", "human", "rgb_array"):
        raise ValueError("render_mode must be None, 'ansi', 'human', or 'rgb_array'.")
    if int(render_window_size) <= 0:
        raise ValueError("render_window_size must be positive.")


__all__ = ["CleanUpRenderer", "validate_renderer_options"]
