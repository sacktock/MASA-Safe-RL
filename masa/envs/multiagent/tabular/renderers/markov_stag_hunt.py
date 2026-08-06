from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
Color = tuple[int, int, int]

MEADOW = (220, 218, 188)
MEADOW_ALT = (210, 211, 176)
GRID = (170, 169, 139)
PLANT = (72, 148, 83)
PLANT_LIGHT = (128, 184, 91)
PLANT_DARK = (48, 105, 61)
STAG = (132, 79, 48)
STAG_LIGHT = (186, 128, 75)
STAG_DARK = (80, 49, 36)
IVORY = (242, 231, 195)
SHADOW = (124, 119, 94)
WHITE = (250, 245, 226)
MAUL = (196, 55, 48)
PLAYER_COLORS: tuple[Color, ...] = (
    (48, 108, 184),
    (214, 105, 45),
    (137, 83, 171),
    (38, 151, 137),
    (203, 69, 111),
    (174, 146, 35),
)


class MarkovStagHuntEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    grid_size: tuple[int, int]
    agent_positions: dict[str, Position]
    plants: set[Position]
    stags: set[Position]
    _last_mauled_agents: set[str]


class MarkovStagHuntRenderer:
    def __init__(self, env: MarkovStagHuntEnv) -> None:
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
        rows, cols = self.env.grid_size
        chars = [["." for _ in range(cols)] for _ in range(rows)]
        for row, col in self.env.plants:
            chars[row][col] = "p"
        for row, col in self.env.stags:
            chars[row][col] = "S"
        cells: dict[Position, list[str]] = defaultdict(list)
        for agent, position in self.env.agent_positions.items():
            cells[position].append(agent)
        for (row, col), agents in cells.items():
            chars[row][col] = str(len(agents)) if len(agents) > 1 else agents[0].split("_")[-1]
        return "\n".join(" ".join(row) for row in chars)

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        if not pygame.font.get_init():
            pygame.font.init()
        rows, cols = self.env.grid_size
        cell = max(48, self.env.render_window_size // max(rows, cols))
        width, height = cols * cell, rows * cell
        surface = pygame.Surface((width, height))

        for row in range(rows):
            for col in range(cols):
                rect = pygame.Rect(col * cell, row * cell, cell, cell)
                color = MEADOW if (row + col) % 2 == 0 else MEADOW_ALT
                pygame.draw.rect(surface, color, rect)
                self._draw_ground_texture(surface, rect, cell, row, col)
                pygame.draw.rect(surface, GRID, rect, width=max(1, cell // 50))

        for position in self.env.plants:
            self._draw_plant(surface, position, cell)
        for position in self.env.stags:
            self._draw_stag(surface, position, cell)

        cells: dict[Position, list[str]] = defaultdict(list)
        for agent, position in self.env.agent_positions.items():
            cells[position].append(agent)
        for position, agents in cells.items():
            for slot, agent in enumerate(sorted(agents)):
                self._draw_agent(surface, position, agent, slot, len(agents), cell)

        return np.ascontiguousarray(
            np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)), dtype=np.uint8
        )

    def _render_human(self, frame: np.ndarray) -> None:
        if self._window_closed:
            return
        import pygame

        if self._window is None:
            pygame.init()
            pygame.display.set_caption("MASA - Markov Stag Hunt")
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
    def _draw_ground_texture(surface: Any, rect: Any, cell: int, row: int, col: int) -> None:
        import pygame

        radius = max(1, cell // 60)
        for index in range(3):
            x = rect.left + ((row * 17 + col * 29 + index * 31) % max(1, cell - 10)) + 5
            y = rect.top + ((row * 23 + col * 11 + index * 19) % max(1, cell - 10)) + 5
            pygame.draw.circle(surface, GRID, (x, y), radius)

    @staticmethod
    def _center(position: Position, cell: int) -> Position:
        row, col = position
        return col * cell + cell // 2, row * cell + cell // 2

    def _draw_plant(self, surface: Any, position: Position, cell: int) -> None:
        import pygame

        cx, cy = self._center(position, cell)
        radius = max(9, cell // 7)
        pygame.draw.ellipse(
            surface,
            SHADOW,
            pygame.Rect(cx - radius, cy + radius // 2, radius * 2, radius),
        )
        pygame.draw.line(surface, PLANT_DARK, (cx, cy + radius), (cx, cy - radius), max(2, cell // 28))
        for dx, dy in ((-radius, -radius // 2), (radius, -radius // 3), (-radius // 2, radius // 3), (radius // 2, radius // 2)):
            pygame.draw.ellipse(
                surface,
                PLANT,
                pygame.Rect(cx + dx - radius // 2, cy + dy - radius // 3, radius, 2 * radius // 3),
            )
        pygame.draw.circle(surface, PLANT_LIGHT, (cx, cy - radius), max(4, radius // 2))

    def _draw_stag(self, surface: Any, position: Position, cell: int) -> None:
        import pygame

        cx, cy = self._center(position, cell)
        body_w, body_h = int(cell * 0.48), int(cell * 0.27)
        body = pygame.Rect(
            cx - int(cell * 0.28),
            cy - int(cell * 0.03),
            body_w,
            body_h,
        )
        pygame.draw.ellipse(
            surface,
            SHADOW,
            pygame.Rect(
                body.left + int(cell * 0.03),
                body.bottom - int(cell * 0.01),
                body_w,
                max(4, body_h // 3),
            ),
        )
        pygame.draw.ellipse(surface, STAG, body)

        neck = [
            (cx + int(cell * 0.08), cy + int(cell * 0.02)),
            (cx + int(cell * 0.15), cy - int(cell * 0.24)),
            (cx + int(cell * 0.27), cy - int(cell * 0.20)),
            (cx + int(cell * 0.21), cy + int(cell * 0.08)),
        ]
        pygame.draw.polygon(surface, STAG, neck)

        head_rect = pygame.Rect(
            cx + int(cell * 0.13),
            cy - int(cell * 0.31),
            int(cell * 0.24),
            int(cell * 0.16),
        )
        pygame.draw.ellipse(surface, STAG_LIGHT, head_rect)
        muzzle = pygame.Rect(
            head_rect.right - int(cell * 0.04),
            head_rect.centery,
            int(cell * 0.11),
            int(cell * 0.08),
        )
        pygame.draw.ellipse(surface, IVORY, muzzle)
        pygame.draw.circle(
            surface,
            STAG_DARK,
            (muzzle.right - max(2, cell // 45), muzzle.centery),
            max(2, cell // 50),
        )

        ear_y = head_rect.top + int(cell * 0.01)
        pygame.draw.polygon(
            surface,
            STAG_LIGHT,
            [
                (head_rect.left + int(cell * 0.04), ear_y + int(cell * 0.03)),
                (head_rect.left - int(cell * 0.03), ear_y - int(cell * 0.07)),
                (head_rect.left + int(cell * 0.09), ear_y),
            ],
        )
        pygame.draw.polygon(
            surface,
            STAG_LIGHT,
            [
                (head_rect.centerx, ear_y + int(cell * 0.02)),
                (head_rect.centerx + int(cell * 0.03), ear_y - int(cell * 0.08)),
                (head_rect.centerx + int(cell * 0.08), ear_y + int(cell * 0.02)),
            ],
        )

        eye = (head_rect.right - int(cell * 0.08), head_rect.top + int(cell * 0.05))
        pygame.draw.circle(surface, STAG_DARK, eye, max(2, cell // 50))
        pygame.draw.circle(surface, WHITE, (eye[0] + 1, eye[1] - 1), 1)

        leg_width = max(2, cell // 35)
        hoof_y = cy + int(cell * 0.32)
        for leg_x in (body.left + int(cell * 0.10), body.right - int(cell * 0.10)):
            knee = (leg_x + int(cell * 0.02), cy + int(cell * 0.20))
            pygame.draw.line(surface, STAG_DARK, (leg_x, body.bottom - 2), knee, leg_width)
            pygame.draw.line(surface, STAG_DARK, knee, (leg_x, hoof_y), leg_width)
            pygame.draw.line(
                surface,
                STAG_DARK,
                (leg_x - int(cell * 0.025), hoof_y),
                (leg_x + int(cell * 0.025), hoof_y),
                leg_width,
            )

        tail = [
            (body.left + int(cell * 0.03), body.top + int(cell * 0.05)),
            (body.left - int(cell * 0.10), body.top - int(cell * 0.02)),
            (body.left + int(cell * 0.01), body.top + int(cell * 0.13)),
        ]
        pygame.draw.polygon(surface, STAG_LIGHT, tail)

        for offset_x, offset_y in ((-0.15, 0.03), (-0.05, 0.08), (0.05, 0.02)):
            pygame.draw.circle(
                surface,
                STAG_LIGHT,
                (cx + int(cell * offset_x), cy + int(cell * offset_y)),
                max(2, cell // 45),
            )

        antler_width = max(2, cell // 45)
        for root_x in (head_rect.left + int(cell * 0.07), head_rect.centerx):
            root = (root_x, head_rect.top + int(cell * 0.01))
            mid = (root_x - int(cell * 0.04), head_rect.top - int(cell * 0.07))
            tip = (root_x - int(cell * 0.01), head_rect.top - int(cell * 0.14))
            pygame.draw.lines(surface, IVORY, False, [root, mid, tip], antler_width)
            pygame.draw.line(
                surface,
                IVORY,
                mid,
                (mid[0] - int(cell * 0.08), mid[1] - int(cell * 0.04)),
                antler_width,
            )
            pygame.draw.line(
                surface,
                IVORY,
                tip,
                (tip[0] + int(cell * 0.07), tip[1] - int(cell * 0.04)),
                antler_width,
            )

    def _draw_agent(
        self,
        surface: Any,
        position: Position,
        agent: str,
        slot: int,
        count: int,
        cell: int,
    ) -> None:
        import pygame

        cx, cy = self._center(position, cell)
        radius = max(12, cell // (5 if count == 1 else 7))
        if count > 1:
            spacing = radius * 2
            cx += int((slot - (count - 1) / 2) * spacing)
        index = int(agent.split("_")[-1])
        color = PLAYER_COLORS[index % len(PLAYER_COLORS)]
        pygame.draw.circle(surface, SHADOW, (cx + 3, cy + 4), radius + 3)
        if agent in self.env._last_mauled_agents:
            pygame.draw.circle(surface, MAUL, (cx, cy), radius + max(5, cell // 25))
        pygame.draw.circle(surface, WHITE, (cx, cy), radius + max(3, cell // 35))
        pygame.draw.circle(surface, color, (cx, cy), radius)
        font = pygame.font.Font(None, max(18, int(radius * 1.2)))
        label = font.render(str(index), True, WHITE)
        surface.blit(label, label.get_rect(center=(cx, cy)))


def validate_renderer_options(render_mode: str | None, render_window_size: int) -> None:
    if render_mode not in (None, "ansi", "human", "rgb_array"):
        raise ValueError("render_mode must be None, 'ansi', 'human', or 'rgb_array'.")
    if int(render_window_size) <= 0:
        raise ValueError("render_window_size must be positive.")


__all__ = ["MarkovStagHuntRenderer", "validate_renderer_options"]
