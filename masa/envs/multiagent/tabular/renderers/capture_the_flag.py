from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
Color = tuple[int, int, int]

FLOOR: Color = (176, 171, 158)
FLOOR_DARK: Color = (160, 155, 143)
RED_GROUND: Color = (160, 22, 35)
BLUE_GROUND: Color = (35, 25, 160)
RED: Color = (225, 55, 85)
BLUE: Color = (85, 70, 225)
WALL: Color = (92, 94, 98)
DAMAGED_WALL: Color = (65, 66, 70)
RUBBLE: Color = (38, 38, 42)
INDICATOR: Color = (107, 63, 160)
GOLD: Color = (218, 165, 32)
WHITE: Color = (245, 245, 235)
BEAM: Color = (252, 244, 90)
BLACK: Color = (10, 10, 12)


class CaptureTheFlagEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    _rows: int
    _cols: int
    _grid: list[str]
    _solid_walls: set[Position]
    _destroyable_walls: dict[Position, Any]
    _terrain: dict[Position, str]
    _indicator_tiles: set[Position]
    _flags: dict[str, Any]
    _avatars: dict[str, Any]
    _last_zap_cells: set[Position]


class CaptureTheFlagRenderer:
    def __init__(self, env: CaptureTheFlagEnv) -> None:
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
        for position, wall in self.env._destroyable_walls.items():
            x, y = position
            chars[y][x] = "D" if wall.active else "."
        for flag in self.env._flags.values():
            if flag.carried_by is None:
                x, y = flag.position
                chars[y][x] = "R" if flag.team == "red" else "B"
        for avatar in self.env._avatars.values():
            if avatar.active:
                x, y = avatar.position
                chars[y][x] = "r" if avatar.team == "red" else "b"
        return "\n".join("".join(row) for row in chars)

    def _render_rgb_array(self) -> np.ndarray:
        cell = max(4, self.env.render_window_size // max(self.env._rows, self.env._cols))
        frame = np.empty((self.env._rows * cell, self.env._cols * cell, 3), dtype=np.uint8)
        frame[:] = BLACK

        for y in range(self.env._rows):
            for x in range(self.env._cols):
                position = (x, y)
                color = FLOOR if (x + y) % 2 == 0 else FLOOR_DARK
                if position in self.env._solid_walls:
                    color = WALL
                wall = self.env._destroyable_walls.get(position)
                if wall is not None:
                    color = WALL if wall.health > 2 else DAMAGED_WALL if wall.active else RUBBLE
                terrain = self.env._terrain.get(position)
                if terrain == "red":
                    color = RED_GROUND
                elif terrain == "blue":
                    color = BLUE_GROUND
                if position in self.env._indicator_tiles:
                    color = INDICATOR
                self._fill(frame, position, cell, color)

        for flag in self.env._flags.values():
            self._outline(frame, flag.home, cell, GOLD, max(1, cell // 10))
            if flag.carried_by is None:
                self._draw_flag(frame, flag.position, flag.team, cell)
        for position in self.env._last_zap_cells:
            self._draw_beam(frame, position, cell)
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
            pygame.display.set_caption("MASA - Capture the Flag")
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

    @staticmethod
    def _outline(
        frame: np.ndarray,
        position: Position,
        cell: int,
        color: Color,
        width: int,
    ) -> None:
        x, y = position
        y0, x0 = y * cell, x * cell
        frame[y0 : y0 + width, x0 : x0 + cell] = color
        frame[y0 + cell - width : y0 + cell, x0 : x0 + cell] = color
        frame[y0 : y0 + cell, x0 : x0 + width] = color
        frame[y0 : y0 + cell, x0 + cell - width : x0 + cell] = color

    def _draw_flag(self, frame: np.ndarray, position: Position, team: str, cell: int) -> None:
        x, y = position
        y0, x0 = y * cell, x * cell
        mid = cell // 2
        color = RED if team == "red" else BLUE
        frame[y0 + cell // 5 : y0 + 4 * cell // 5, x0 + mid : x0 + mid + 2] = WHITE
        frame[y0 + cell // 5 : y0 + cell // 2, x0 + mid : x0 + 4 * cell // 5] = color

    def _draw_beam(self, frame: np.ndarray, position: Position, cell: int) -> None:
        x, y = position
        y0, x0 = y * cell, x * cell
        mid = cell // 2
        frame[y0 : y0 + cell, x0 + mid : x0 + mid + 1] = BEAM
        frame[y0 + mid : y0 + mid + 1, x0 : x0 + cell] = BEAM

    def _draw_avatar(self, frame: np.ndarray, avatar: Any, cell: int) -> None:
        x, y = avatar.position
        y0, x0 = y * cell, x * cell
        pad = max(2, cell // 6)
        color = np.asarray(RED if avatar.team == "red" else BLUE, dtype=np.int16)
        color = tuple(np.clip(color + (avatar.health - 2) * 35, 0, 255).astype(np.uint8))
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
        if avatar.carrying_flag is not None:
            marker = RED if avatar.carrying_flag == "red" else BLUE
            frame[y0 + pad : y0 + 2 * pad, x0 + pad : x0 + 2 * pad] = marker


def validate_renderer_options(render_mode: str | None, render_window_size: int) -> None:
    if render_mode not in (None, "ansi", "human", "rgb_array"):
        raise ValueError("render_mode must be None, 'ansi', 'human', or 'rgb_array'.")
    if int(render_window_size) <= 0:
        raise ValueError("render_window_size must be positive.")


__all__ = ["CaptureTheFlagRenderer", "validate_renderer_options"]
