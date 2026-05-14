from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np

from masa.envs.tabular.renderers.pacman import (
    PacmanRenderer as _BasePacmanRenderer,
    PacmanHat,
    RGBColor,
    _GhostSnapshot,
    _PacmanSnapshot,
    validate_renderer_options,
)


class CoinPacmanEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    window_size: int
    pacman_hat: PacmanHat
    ghost_colors: tuple[RGBColor, ...] | None
    _layout: np.ndarray
    _n_row: int
    _n_col: int
    _state: int | None
    _start_state: int
    _step_count: int
    _coin_array: np.ndarray | None
    _agent_term_x: int
    _agent_term_y: int
    _reverse_state_map: dict[int, tuple[int, int, int, int, int, int, int]]


class PacmanWithCoinsRenderer(_BasePacmanRenderer):
    """Festival-style renderer for spatial Pacman-with-coins variants."""

    def __init__(self, env: CoinPacmanEnv) -> None:
        super().__init__(env)
        self.env = env

    def _snapshot(self) -> _PacmanSnapshot:
        state = self.env._start_state if self.env._state is None else int(self.env._state)
        agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, _ = self.env._reverse_state_map[state]
        if self.env._coin_array is None:
            collectibles = np.zeros_like(self.env._layout, dtype=np.float32)
        else:
            collectibles = np.asarray(self.env._coin_array, dtype=np.float32).copy()
            collectibles[self.env._layout == 1] = 0.0

        return _PacmanSnapshot(
            layout=self.env._layout,
            collectibles=collectibles,
            terminal=(self.env._agent_term_y, self.env._agent_term_x),
            agent_position=(agent_y, agent_x),
            agent_direction=agent_direction,
            ghosts=(_GhostSnapshot((ghost_y, ghost_x), ghost_direction),),
            step_count=self.env._step_count,
        )

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        snapshot = self._snapshot()
        cell_size = max(12, int(self.env.window_size) // max(snapshot.layout.shape))
        scale = 3
        high_cell = cell_size * scale
        high_size = (snapshot.layout.shape[1] * high_cell, snapshot.layout.shape[0] * high_cell)
        surface = pygame.Surface(high_size)

        for row in range(snapshot.layout.shape[0]):
            for col in range(snapshot.layout.shape[1]):
                rect = pygame.Rect(col * high_cell, row * high_cell, high_cell, high_cell)
                if snapshot.layout[row, col] == 1:
                    self._draw_wall_tile(surface, rect, high_cell)
                else:
                    self._draw_floor_tile(surface, rect, high_cell)

        self._draw_terminal(surface, high_cell, snapshot.terminal)
        self._draw_collectibles(surface, high_cell, snapshot.collectibles, "coins")
        self._draw_ghosts(surface, high_cell, snapshot.ghosts, snapshot.agent_position)
        self._draw_pacman(surface, high_cell, snapshot.agent_position, snapshot.agent_direction, snapshot.step_count)

        if scale > 1:
            final_size = (snapshot.layout.shape[1] * cell_size, snapshot.layout.shape[0] * cell_size)
            surface = pygame.transform.smoothscale(surface, final_size)

        frame = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        return np.ascontiguousarray(frame, dtype=np.uint8)


__all__ = ["PacmanWithCoinsRenderer", "validate_renderer_options"]

