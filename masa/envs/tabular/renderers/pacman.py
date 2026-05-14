from __future__ import annotations

import math
import os
from typing import Any, Literal, Protocol, Sequence

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
RGBColor = tuple[int, int, int]
PacmanHat = Literal["none", "cap", "crown", "wizard"]

LEFT = 0
RIGHT = 1
DOWN = 2
UP = 3

FLOOR_COLOR = (14, 16, 28)
FLOOR_GRID_COLOR = (22, 27, 46)
WALL_COLOR = (24, 84, 218)
WALL_HIGHLIGHT_COLOR = (77, 167, 255)
WALL_SHADOW_COLOR = (9, 26, 92)
PELLET_COLOR = (255, 223, 125)
POWER_PELLET_COLOR = (255, 242, 177)
PACMAN_COLOR = (255, 217, 59)
PACMAN_SHADOW_COLOR = (211, 148, 28)
PORTAL_COLOR = (96, 232, 165)
PORTAL_DARK_COLOR = (21, 102, 88)
COLLISION_COLOR = (255, 77, 109)
GHOST_COLORS = (
    (238, 63, 88),
    (71, 202, 255),
    (255, 151, 60),
    (192, 115, 255),
    (80, 220, 154),
    (255, 111, 196),
)

DIRECTION_DELTAS: dict[int, Position] = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    UP: (-1, 0),
}


class TabularPacmanEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    pacman_hat: PacmanHat
    ghost_colors: tuple[RGBColor, ...] | None
    _layout: np.ndarray
    _n_row: int
    _n_col: int
    _state: int | None
    _start_state: int
    _step_count: int
    _food_x: int
    _food_y: int
    _agent_term_x: int
    _agent_term_y: int
    _reverse_state_map: dict[int, tuple[int, int, int, int, int, int, int]]


class PacmanRenderer:
    """Festival-style renderer for tabular Pacman variants."""

    def __init__(self, env: TabularPacmanEnv) -> None:
        self.env = env
        self._ghost_colors = _normalize_ghost_colors(env.ghost_colors)
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
        chars = np.full(snapshot.layout.shape, " ", dtype="<U1")
        chars[snapshot.layout == 1] = "#"
        chars[snapshot.collectibles > 0.0] = "."
        chars[snapshot.terminal] = "T"
        for ghost in snapshot.ghosts:
            chars[ghost.position] = "G"
        chars[snapshot.agent_position] = "P"
        return "\n".join("".join(row) for row in chars)

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        snapshot = self._snapshot()
        cell_size = max(12, int(self.env.render_window_size) // max(snapshot.layout.shape))
        scale = 3
        high_cell = cell_size * scale
        high_size = (snapshot.layout.shape[1] * high_cell, snapshot.layout.shape[0] * high_cell)
        surface = pygame.Surface(high_size)
        surface.fill(FLOOR_COLOR)

        for row in range(snapshot.layout.shape[0]):
            for col in range(snapshot.layout.shape[1]):
                rect = pygame.Rect(col * high_cell, row * high_cell, high_cell, high_cell)
                if snapshot.layout[row, col] == 1:
                    self._draw_wall_tile(surface, rect, high_cell)
                else:
                    self._draw_floor_tile(surface, rect, high_cell)

        self._draw_terminal(surface, high_cell, snapshot.terminal)
        self._draw_collectibles(surface, high_cell, snapshot.collectibles, "food")
        self._draw_ghosts(surface, high_cell, snapshot.ghosts, snapshot.agent_position)
        self._draw_pacman(surface, high_cell, snapshot.agent_position, snapshot.agent_direction, snapshot.step_count)

        if scale > 1:
            final_size = (snapshot.layout.shape[1] * cell_size, snapshot.layout.shape[0] * cell_size)
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
            pygame.display.set_caption("MASA - Pacman")
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

    def _snapshot(self) -> "_PacmanSnapshot":
        state = self.env._start_state if self.env._state is None else int(self.env._state)
        agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food = self.env._reverse_state_map[state]
        collectibles = np.zeros_like(self.env._layout, dtype=np.float32)
        if food:
            collectibles[self.env._food_y, self.env._food_x] = 1.0
        return _PacmanSnapshot(
            layout=self.env._layout,
            collectibles=collectibles,
            terminal=(self.env._agent_term_y, self.env._agent_term_x),
            agent_position=(agent_y, agent_x),
            agent_direction=agent_direction,
            ghosts=(_GhostSnapshot((ghost_y, ghost_x), ghost_direction),),
            step_count=self.env._step_count,
        )

    def _draw_floor_tile(self, surface: Any, rect: Any, cell_size: int) -> None:
        import pygame

        inset = max(1, cell_size // 24)
        inner = rect.inflate(-inset * 2, -inset * 2)
        pygame.draw.rect(surface, FLOOR_COLOR, rect)
        pygame.draw.rect(surface, FLOOR_GRID_COLOR, inner, width=max(1, cell_size // 40), border_radius=max(2, cell_size // 12))

    def _draw_wall_tile(self, surface: Any, rect: Any, cell_size: int) -> None:
        import pygame

        radius = max(4, cell_size // 6)
        pygame.draw.rect(surface, WALL_SHADOW_COLOR, rect, border_radius=radius)
        body = rect.inflate(-max(2, cell_size // 14), -max(2, cell_size // 14))
        pygame.draw.rect(surface, WALL_COLOR, body, border_radius=radius)
        highlight = body.inflate(-max(3, cell_size // 8), -max(3, cell_size // 8))
        pygame.draw.rect(surface, WALL_HIGHLIGHT_COLOR, highlight, width=max(1, cell_size // 24), border_radius=radius)

    def _draw_terminal(self, surface: Any, cell_size: int, terminal: Position) -> None:
        import pygame

        center = _cell_center(terminal[0], terminal[1], cell_size)
        radius = max(6, cell_size // 4)
        pygame.draw.circle(surface, PORTAL_DARK_COLOR, center, radius)
        pygame.draw.circle(surface, PORTAL_COLOR, center, radius, width=max(2, cell_size // 18))
        pygame.draw.circle(surface, (173, 255, 216), center, max(2, radius // 3))

    def _draw_collectibles(self, surface: Any, cell_size: int, collectibles: np.ndarray, mode: str) -> None:
        import pygame

        cells = np.argwhere(collectibles > 0.0)
        for row, col in cells:
            center = _cell_center(int(row), int(col), cell_size)
            if mode == "food":
                glow = max(5, cell_size // 6)
                radius = max(3, cell_size // 11)
                pygame.draw.circle(surface, (108, 79, 31), center, glow)
                pygame.draw.circle(surface, POWER_PELLET_COLOR, center, radius)
            else:
                pygame.draw.circle(surface, PELLET_COLOR, center, max(2, cell_size // 16))

    def _draw_ghosts(
        self,
        surface: Any,
        cell_size: int,
        ghosts: tuple["_GhostSnapshot", ...],
        agent_position: Position,
    ) -> None:
        import pygame

        ghosts_by_position: dict[Position, list[tuple[int, _GhostSnapshot]]] = {}
        for idx, ghost in enumerate(ghosts):
            ghosts_by_position.setdefault(ghost.position, []).append((idx, ghost))

        collision_positions = {ghost.position for ghost in ghosts if ghost.position == agent_position}
        for position, positioned_ghosts in ghosts_by_position.items():
            for offset_idx, (ghost_idx, ghost) in enumerate(positioned_ghosts):
                offset = _overlap_offset(offset_idx, cell_size)
                center = _offset_position(_cell_center(position[0], position[1], cell_size), offset)
                color = self._ghost_colors[ghost_idx % len(self._ghost_colors)]
                if position in collision_positions:
                    pygame.draw.circle(surface, COLLISION_COLOR, center, max(7, cell_size // 3), width=max(3, cell_size // 18))
                self._draw_single_ghost(surface, center, cell_size, color, ghost.direction)

    def _draw_single_ghost(self, surface: Any, center: Position, cell_size: int, color: RGBColor, direction: int) -> None:
        import pygame

        width = int(cell_size * 0.58)
        height = int(cell_size * 0.68)
        left = center[0] - width // 2
        top = center[1] - height // 2
        body_rect = pygame.Rect(left, top + height // 5, width, int(height * 0.72))
        head_center = (center[0], top + height // 3)
        head_radius = width // 2

        pygame.draw.circle(surface, color, head_center, head_radius)
        pygame.draw.rect(surface, color, body_rect)
        scallop_radius = max(3, width // 7)
        for x_pos in (left + scallop_radius, center[0], left + width - scallop_radius):
            pygame.draw.circle(surface, FLOOR_COLOR, (x_pos, top + height), scallop_radius)

        shadow = (max(0, color[0] - 70), max(0, color[1] - 70), max(0, color[2] - 70))
        pygame.draw.arc(surface, shadow, pygame.Rect(left, top + height // 5, width, height // 2), math.pi, 2 * math.pi, max(2, cell_size // 20))

        eye_radius = max(3, cell_size // 13)
        pupil_radius = max(1, eye_radius // 2)
        eye_y = top + height // 3
        eye_dx = width // 5
        look = DIRECTION_DELTAS.get(direction, (0, 0))
        pupil_offset = (look[1] * max(1, eye_radius // 3), look[0] * max(1, eye_radius // 3))
        for eye_x in (center[0] - eye_dx, center[0] + eye_dx):
            pygame.draw.circle(surface, (248, 250, 255), (eye_x, eye_y), eye_radius)
            pygame.draw.circle(surface, (20, 34, 78), (eye_x + pupil_offset[0], eye_y + pupil_offset[1]), pupil_radius)

    def _draw_pacman(self, surface: Any, cell_size: int, position: Position, direction: int, step_count: int) -> None:
        import pygame

        center = _cell_center(position[0], position[1], cell_size)
        radius = int(cell_size * 0.36)
        pygame.draw.circle(surface, PACMAN_SHADOW_COLOR, (center[0] + max(1, cell_size // 24), center[1] + max(1, cell_size // 24)), radius)
        pygame.draw.circle(surface, PACMAN_COLOR, center, radius)

        mouth_open = math.radians(34 + (step_count % 2) * 10)
        angle = _direction_angle(direction)
        mouth_margin = max(2, cell_size // 28)
        mouth_extent = int(math.ceil((radius + mouth_margin) / math.cos(mouth_open)))
        mouth_points = [
            center,
            (
                int(center[0] + math.cos(angle - mouth_open) * mouth_extent),
                int(center[1] + math.sin(angle - mouth_open) * mouth_extent),
            ),
            (
                int(center[0] + math.cos(angle + mouth_open) * mouth_extent),
                int(center[1] + math.sin(angle + mouth_open) * mouth_extent),
            ),
        ]
        pygame.draw.polygon(surface, FLOOR_COLOR, mouth_points)

        eye_angle = _pacman_eye_angle(direction)
        eye_pos = (
            int(center[0] + math.cos(eye_angle) * radius * 0.42),
            int(center[1] + math.sin(eye_angle) * radius * 0.42),
        )
        pygame.draw.circle(surface, (25, 21, 28), eye_pos, max(2, cell_size // 24))
        self._draw_pacman_hat(surface, center, radius, cell_size)

    def _draw_pacman_hat(self, surface: Any, center: Position, radius: int, cell_size: int) -> None:
        import pygame

        hat = self.env.pacman_hat
        if hat == "none":
            return

        x, y = center
        top = y - radius
        outline = (40, 33, 46)

        if hat == "cap":
            brim = pygame.Rect(x - radius, top + radius // 5, radius * 2, max(3, cell_size // 12))
            crown = pygame.Rect(x - radius // 2, top - radius // 4, radius, radius // 2)
            pygame.draw.rect(surface, outline, brim.inflate(max(2, cell_size // 36), max(2, cell_size // 36)), border_radius=max(2, cell_size // 20))
            pygame.draw.ellipse(surface, (255, 82, 99), crown)
            pygame.draw.rect(surface, (201, 43, 73), brim, border_radius=max(2, cell_size // 20))
            pygame.draw.circle(surface, (255, 179, 92), (x + radius // 4, top), max(2, cell_size // 28))
            return

        if hat == "crown":
            base_y = top + radius // 4
            crown_points = [
                (x - radius, base_y),
                (x - radius * 3 // 4, top - radius // 3),
                (x - radius // 4, base_y),
                (x, top - radius // 2),
                (x + radius // 4, base_y),
                (x + radius * 3 // 4, top - radius // 3),
                (x + radius, base_y),
                (x + radius, base_y + max(4, cell_size // 8)),
                (x - radius, base_y + max(4, cell_size // 8)),
            ]
            pygame.draw.polygon(surface, outline, crown_points)
            inset = max(2, cell_size // 30)
            inner_points = [(px, py + inset) for px, py in crown_points[:7]] + [
                (x + radius - inset, base_y + max(4, cell_size // 8) - inset),
                (x - radius + inset, base_y + max(4, cell_size // 8) - inset),
            ]
            pygame.draw.polygon(surface, (255, 209, 82), inner_points)
            pygame.draw.circle(surface, (78, 201, 255), (x, top - radius // 5), max(2, cell_size // 30))
            return

        cone_points = [
            (x, top - radius // 2),
            (x - radius, top + radius // 2),
            (x + radius, top + radius // 2),
        ]
        pygame.draw.polygon(surface, outline, cone_points)
        pygame.draw.polygon(surface, (116, 82, 255), [(x, top - radius // 3), (x - radius + 3, top + radius // 2), (x + radius - 3, top + radius // 2)])
        brim = pygame.Rect(x - radius, top + radius // 3, radius * 2, max(4, cell_size // 10))
        pygame.draw.rect(surface, outline, brim.inflate(max(2, cell_size // 36), max(2, cell_size // 36)), border_radius=max(2, cell_size // 20))
        pygame.draw.rect(surface, (77, 58, 180), brim, border_radius=max(2, cell_size // 20))
        pygame.draw.circle(surface, (255, 245, 160), (x + radius // 4, top), max(2, cell_size // 28))


class _GhostSnapshot:
    def __init__(self, position: Position, direction: int) -> None:
        self.position = position
        self.direction = direction


class _PacmanSnapshot:
    def __init__(
        self,
        *,
        layout: np.ndarray,
        collectibles: np.ndarray,
        terminal: Position,
        agent_position: Position,
        agent_direction: int,
        ghosts: tuple[_GhostSnapshot, ...],
        step_count: int,
    ) -> None:
        self.layout = layout
        self.collectibles = collectibles
        self.terminal = terminal
        self.agent_position = agent_position
        self.agent_direction = agent_direction
        self.ghosts = ghosts
        self.step_count = step_count


def validate_renderer_options(render_mode: str | None, render_window_size: int, pacman_hat: str) -> None:
    if render_mode not in (None, "ansi", "rgb_array", "human"):
        raise ValueError("render_mode must be None, 'ansi', 'rgb_array', or 'human'.")
    if int(render_window_size) <= 0:
        raise ValueError("render_window_size must be positive.")
    if pacman_hat not in ("none", "cap", "crown", "wizard"):
        raise ValueError("pacman_hat must be 'none', 'cap', 'crown', or 'wizard'.")


def _normalize_ghost_colors(colors: Sequence[Sequence[int]] | None) -> tuple[RGBColor, ...]:
    if colors is None:
        return GHOST_COLORS

    normalized: list[RGBColor] = []
    for color in colors:
        if len(color) != 3:
            raise ValueError("Each ghost color must be an RGB triplet.")
        red, green, blue = (int(channel) for channel in color)
        if not all(0 <= channel <= 255 for channel in (red, green, blue)):
            raise ValueError("Ghost color channels must be in [0, 255].")
        normalized.append((red, green, blue))

    if not normalized:
        raise ValueError("ghost_colors must contain at least one RGB triplet when provided.")
    return tuple(normalized)


def _cell_center(row: int, col: int, cell_size: int) -> Position:
    return col * cell_size + cell_size // 2, row * cell_size + cell_size // 2


def _offset_position(position: Position, offset: Position) -> Position:
    return position[0] + offset[0], position[1] + offset[1]


def _overlap_offset(index: int, cell_size: int) -> Position:
    offsets = (
        (0.0, 0.0),
        (-0.16, -0.16),
        (0.16, -0.16),
        (-0.16, 0.16),
        (0.16, 0.16),
        (0.0, -0.22),
        (0.0, 0.22),
    )
    x, y = offsets[index % len(offsets)]
    return int(cell_size * x), int(cell_size * y)


def _direction_angle(direction: int) -> float:
    return {
        LEFT: math.pi,
        RIGHT: 0.0,
        DOWN: math.pi / 2.0,
        UP: -math.pi / 2.0,
    }.get(direction, 0.0)


def _pacman_eye_angle(direction: int) -> float:
    if direction == LEFT:
        return math.pi + math.pi / 2.4
    if direction == UP:
        return -math.pi / 2.0 - math.pi / 2.4
    return -math.pi / 2.4


def _fit_size(source: Position, target: Position) -> Position:
    source_width, source_height = source
    target_width, target_height = target
    scale = min(target_width / source_width, target_height / source_height)
    return max(1, int(source_width * scale)), max(1, int(source_height * scale))
