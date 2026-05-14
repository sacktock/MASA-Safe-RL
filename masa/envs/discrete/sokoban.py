from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from masa.envs.discrete.base import DiscreteEnv

SIZE_X = 6
SIZE_Y = 6

START_AGENT_LOCATION = (2, 1)
START_BOX_LOCATION = (2, 2)
TARGET_LOCATION = (4, 4)

WALLS = (
    {(x, 0) for x in range(6)}
    | {(0, y) for y in range(6)}
    | {(x, 5) for x in range(1, 6)}
    | {(5, y) for y in range(1, 5)}
    | {(1, 3), (1, 4), (2, 4), (3, 1), (4, 1)}
)


def calculate_wall_penalty(box_x: int, box_y: int) -> int:
    adjacent = [(box_x + dx, box_y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
    walls_adjacent = [pos in WALLS for pos in adjacent]

    if (
        (walls_adjacent[0] and walls_adjacent[2])
        or (walls_adjacent[0] and walls_adjacent[3])
        or (walls_adjacent[1] and walls_adjacent[2])
        or (walls_adjacent[1] and walls_adjacent[3])
    ):
        return -10

    if any(walls_adjacent):
        return -5

    return 0


def label_fn(obs: int) -> set[str]:
    agent_x, agent_y, box_x, box_y = Sokoban.decode(obs)
    labels = set()

    if (agent_x, agent_y) == TARGET_LOCATION:
        labels.add("goal")

    wall_penalty = calculate_wall_penalty(box_x, box_y)
    if wall_penalty <= -10:
        labels.add("box_corner")
    elif wall_penalty < 0:
        labels.add("box_adjacent_wall")

    return labels


cost_fn = lambda labels: 1.0 if "box_corner" in labels else 0.0


class Sokoban(DiscreteEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, render_mode: str | None = None, render_window_size: int = 512) -> None:
        super().__init__()

        self.size_x = SIZE_X
        self.size_y = SIZE_Y
        self.render_window_size = render_window_size
        self.observation_space = spaces.Discrete(self.size_x * self.size_y * self.size_x * self.size_y)
        self.action_space = spaces.Discrete(4)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._action_to_direction = {
            0: np.array([+1, 0], dtype=int),
            1: np.array([0, -1], dtype=int),
            2: np.array([-1, 0], dtype=int),
            3: np.array([0, +1], dtype=int),
        }

        self._agent_location = np.array(START_AGENT_LOCATION, dtype=int)
        self._target_location = np.array(TARGET_LOCATION, dtype=int)
        self._box_location = np.array(START_BOX_LOCATION, dtype=int)
        self._renderer = SokobanRenderer(self)

    @staticmethod
    def encode(
        agent_x: int,
        agent_y: int,
        box_x: int,
        box_y: int,
        size_x: int = SIZE_X,
        size_y: int = SIZE_Y,
    ) -> int:
        i = agent_x
        i *= size_y
        i += agent_y
        i *= size_x
        i += box_x
        i *= size_y
        i += box_y
        return i

    @staticmethod
    def decode(
        i: int,
        size_x: int = SIZE_X,
        size_y: int = SIZE_Y,
    ) -> tuple[int, int, int, int]:
        box_y = i % size_y
        i //= size_y
        box_x = i % size_x
        i //= size_x
        agent_y = i % size_y
        i //= size_y
        agent_x = i
        return agent_x, agent_y, box_x, box_y

    def _get_obs(self) -> int:
        return self.encode(
            int(self._agent_location[0]),
            int(self._agent_location[1]),
            int(self._box_location[0]),
            int(self._box_location[1]),
        )

    def _get_info(self) -> dict[str, int]:
        return {"box_wall_penalty": calculate_wall_penalty(int(self._box_location[0]), int(self._box_location[1]))}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        super().reset(seed=seed)
        self._agent_location = np.array(START_AGENT_LOCATION, dtype=int)
        self._target_location = np.array(TARGET_LOCATION, dtype=int)
        self._box_location = np.array(START_BOX_LOCATION, dtype=int)
        obs, info = self._get_obs(), self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}!"

        direction = self._action_to_direction[action]
        new_x = int(np.clip(self._agent_location[0] + direction[0], 0, self.size_x - 1))
        new_y = int(np.clip(self._agent_location[1] + direction[1], 0, self.size_y - 1))
        new_agent_location = np.array([new_x, new_y], dtype=int)

        if np.array_equal(new_agent_location, self._box_location):
            new_box_x = int(np.clip(self._box_location[0] + direction[0], 0, self.size_x - 1))
            new_box_y = int(np.clip(self._box_location[1] + direction[1], 0, self.size_y - 1))

            if (new_box_x, new_box_y) not in WALLS:
                self._box_location = np.array([new_box_x, new_box_y], dtype=int)
                self._agent_location = new_agent_location
        elif (new_x, new_y) not in WALLS:
            self._agent_location = new_agent_location

        reward = -1.0
        terminated = np.array_equal(self._agent_location, self._target_location)
        if terminated:
            reward += 50.0

        obs, info = self._get_obs(), self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            return None
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


class SokobanRenderer:
    _BACKGROUND = (241, 239, 232)
    _FLOOR = (224, 215, 195)
    _FLOOR_ALT = (231, 223, 205)
    _GRID = (198, 187, 164)
    _WALL = (92, 76, 60)
    _GOAL = (111, 170, 101)
    _GOAL_CENTER = (235, 247, 227)
    _AGENT = (68, 127, 186)
    _AGENT_RING = (232, 241, 248)
    _BOX = (185, 143, 86)
    _BOX_EDGE = (119, 81, 42)
    _BOX_WARNING = (209, 120, 74)
    _BOX_DANGER = (184, 83, 61)

    def __init__(self, env: Sokoban):
        self.env = env
        self.window = None
        self.clock = None
        self._pygame = None
        self.tile_size = env.render_window_size // max(env.size_x, env.size_y)
        self.grid_width = self.tile_size * env.size_x
        self.grid_height = self.tile_size * env.size_y
        self.x_offset = (env.render_window_size - self.grid_width) // 2
        self.y_offset = (env.render_window_size - self.grid_height) // 2

    def render(self):
        frame = self._draw_frame()
        if self.env.render_mode == "rgb_array":
            return frame
        self._render_human(frame)
        return None

    def close(self) -> None:
        if self._pygame is None:
            return
        if self.window is not None:
            self._pygame.display.quit()
        self._pygame.quit()
        self.window = None
        self.clock = None
        self._pygame = None

    def _draw_frame(self) -> np.ndarray:
        frame = np.full((self.env.render_window_size, self.env.render_window_size, 3), self._BACKGROUND, dtype=np.uint8)

        for x in range(self.env.size_x):
            for y in range(self.env.size_y):
                floor = self._FLOOR if (x + y) % 2 == 0 else self._FLOOR_ALT
                self._fill_cell(frame, x, y, floor)
                self._stroke_cell(frame, x, y, self._GRID, thickness=2)

        for wall in WALLS:
            self._fill_cell(frame, *wall, self._WALL)

        self._draw_goal(frame, *TARGET_LOCATION)
        self._draw_box(
            frame,
            int(self.env._box_location[0]),
            int(self.env._box_location[1]),
            calculate_wall_penalty(int(self.env._box_location[0]), int(self.env._box_location[1])),
        )
        self._draw_agent(frame, int(self.env._agent_location[0]), int(self.env._agent_location[1]))
        return frame

    def _render_human(self, frame: np.ndarray) -> None:
        pygame = self._require_pygame()
        pygame.event.pump()

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.env.metadata["render_fps"])

    def _require_pygame(self):
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise ImportError("pygame is required for render_mode='human'. Install it with `pip install pygame`.") from exc

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.env.render_window_size, self.env.render_window_size))
            pygame.display.set_caption("MASA - Sokoban")
            self.clock = pygame.time.Clock()
            self._pygame = pygame

        return self._pygame

    def _cell_bounds(self, x: int, y: int, padding: int = 0) -> tuple[int, int, int, int]:
        left = self.x_offset + x * self.tile_size + padding
        top = self.y_offset + y * self.tile_size + padding
        right = self.x_offset + (x + 1) * self.tile_size - padding
        bottom = self.y_offset + (y + 1) * self.tile_size - padding
        return left, top, right, bottom

    def _fill_cell(self, frame: np.ndarray, x: int, y: int, color: tuple[int, int, int], padding: int = 1) -> None:
        left, top, right, bottom = self._cell_bounds(x, y, padding)
        frame[top:bottom, left:right] = color

    def _stroke_cell(self, frame: np.ndarray, x: int, y: int, color: tuple[int, int, int], thickness: int = 1) -> None:
        left, top, right, bottom = self._cell_bounds(x, y, 0)
        frame[top:top + thickness, left:right] = color
        frame[bottom - thickness:bottom, left:right] = color
        frame[top:bottom, left:left + thickness] = color
        frame[top:bottom, right - thickness:right] = color

    def _draw_goal(self, frame: np.ndarray, x: int, y: int) -> None:
        self._fill_cell(frame, x, y, self._GOAL, padding=6)
        self._fill_cell(frame, x, y, self._GOAL_CENTER, padding=self.tile_size // 3)

    def _draw_agent(self, frame: np.ndarray, x: int, y: int) -> None:
        left, top, right, bottom = self._cell_bounds(x, y, 0)
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        radius = self.tile_size // 3
        self._draw_circle(frame, cx, cy, radius, self._AGENT_RING)
        self._draw_circle(frame, cx, cy, radius - 6, self._AGENT)

    def _draw_box(self, frame: np.ndarray, x: int, y: int, wall_penalty: int) -> None:
        color = self._BOX
        if wall_penalty <= -10:
            color = self._BOX_DANGER
        elif wall_penalty < 0:
            color = self._BOX_WARNING

        self._fill_cell(frame, x, y, color, padding=self.tile_size // 6)
        left, top, right, bottom = self._cell_bounds(x, y, self.tile_size // 6)
        thickness = max(3, self.tile_size // 16)
        frame[top:top + thickness, left:right] = self._BOX_EDGE
        frame[bottom - thickness:bottom, left:right] = self._BOX_EDGE
        frame[top:bottom, left:left + thickness] = self._BOX_EDGE
        frame[top:bottom, right - thickness:right] = self._BOX_EDGE

    @staticmethod
    def _draw_circle(frame: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        frame[mask] = color
