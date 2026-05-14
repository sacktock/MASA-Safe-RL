from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

Position = tuple[int, int]
RGBColor = tuple[int, int, int]

BACKGROUND_COLOR = (235, 236, 229)
PANEL_COLOR = (226, 229, 222)
TRACK_COLOR = (198, 204, 197)
SLOT_COLOR = (213, 219, 214)
FILLED_COLOR = (74, 139, 214)
FILLED_HIGHLIGHT_COLOR = (132, 185, 239)
EMPTY_COLOR = (198, 70, 76)
START_COLOR = (235, 179, 76)
MARKER_COLOR = (39, 45, 58)
SLOW_COLOR = (107, 161, 105)
FAST_COLOR = (211, 119, 62)
TEXT_COLOR = (39, 45, 58)
MUTED_TEXT_COLOR = (97, 104, 111)
MIN_RENDER_HEIGHT = 160
RENDER_HEIGHT_RATIO = 0.5


class MediaStreamingEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    render_window_size: int
    _state: int | None
    _start_state: int
    _buffer_size: int
    _fast_rate: float
    _slow_rate: float
    _out_rate: float
    _last_action: int | None
    _step_count: int


class MediaStreamingRenderer:
    """Renderer for the tabular Media Streaming buffer MDP."""

    def __init__(self, env: MediaStreamingEnv) -> None:
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
            self._human_window_size = (max(160, int(event.w)), max(96, int(event.h)))
        return True

    def _render_ansi(self) -> str:
        snapshot = self._snapshot()
        slots = ["-" for _ in range(snapshot.buffer_size)]
        slots[0] = "E"
        slots[snapshot.start_state] = "S"
        slots[snapshot.state] = "A"
        action = "none" if snapshot.last_action is None else ("slow" if snapshot.last_action == 0 else "fast")
        return (
            f"buffer: {''.join(slots)}\n"
            f"level: {snapshot.state}/{snapshot.buffer_size - 1}\n"
            f"start: S@{snapshot.start_state}\n"
            "empty: E@0\n"
            f"last_action: {action}"
        )

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        snapshot = self._snapshot()
        width = int(self.env.render_window_size)
        height = max(MIN_RENDER_HEIGHT, int(width * RENDER_HEIGHT_RATIO))
        scale = 3
        surface = pygame.Surface((width * scale, height * scale))
        surface.fill(BACKGROUND_COLOR)

        self._draw_panel(surface, snapshot, width * scale, height * scale)

        if scale > 1:
            surface = pygame.transform.smoothscale(surface, (width, height))

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
            pygame.display.set_caption("MASA - Media Streaming")
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

    def _snapshot(self) -> "_MediaStreamingSnapshot":
        state = int(self.env._start_state if self.env._state is None else self.env._state)
        return _MediaStreamingSnapshot(
            state=state,
            start_state=int(self.env._start_state),
            buffer_size=int(self.env._buffer_size),
            slow_rate=float(self.env._slow_rate),
            fast_rate=float(self.env._fast_rate),
            out_rate=float(self.env._out_rate),
            last_action=self.env._last_action,
            step_count=int(getattr(self.env, "_step_count", 0)),
        )

    def _draw_panel(self, surface: Any, snapshot: "_MediaStreamingSnapshot", width: int, height: int) -> None:
        import pygame

        margin = max(18, width // 18)
        panel = pygame.Rect(margin, margin, width - margin * 2, height - margin * 2)
        pygame.draw.rect(surface, PANEL_COLOR, panel, border_radius=max(8, height // 18))

        title_y = panel.top + max(14, height // 14)
        self._draw_text(surface, "Media Streaming Buffer", (panel.left + margin // 2, title_y), max(16, height // 13), TEXT_COLOR)

        gauge_left = panel.left + margin // 2
        gauge_right = panel.right - margin // 2
        gauge_top = panel.top + int(height * 0.38)
        gauge_height = max(22, height // 8)
        gauge = pygame.Rect(gauge_left, gauge_top, gauge_right - gauge_left, gauge_height)
        pygame.draw.rect(surface, TRACK_COLOR, gauge, border_radius=max(4, gauge_height // 5))

        slot_gap = max(1, width // 220)
        slot_width = max(2, (gauge.width - slot_gap * (snapshot.buffer_size - 1)) // snapshot.buffer_size)
        for idx in range(snapshot.buffer_size):
            left = gauge.left + idx * (slot_width + slot_gap)
            slot = pygame.Rect(left, gauge.top, slot_width, gauge.height)
            color = EMPTY_COLOR if idx == 0 else SLOT_COLOR
            if idx <= snapshot.state:
                color = FILLED_COLOR if idx != 0 else EMPTY_COLOR
            pygame.draw.rect(surface, color, slot, border_radius=max(2, gauge_height // 8))
            if idx <= snapshot.state and idx != 0:
                highlight = slot.inflate(-max(1, slot_width // 4), -max(2, gauge_height // 3))
                pygame.draw.rect(surface, FILLED_HIGHLIGHT_COLOR, highlight, border_radius=max(1, gauge_height // 12))

        self._draw_vertical_marker(surface, gauge, snapshot.start_state, snapshot.buffer_size, START_COLOR, "start")
        self._draw_vertical_marker(surface, gauge, snapshot.state, snapshot.buffer_size, MARKER_COLOR, "now")

        action_color = SLOW_COLOR if snapshot.last_action == 0 else FAST_COLOR if snapshot.last_action == 1 else MUTED_TEXT_COLOR
        action_label = "last: none"
        if snapshot.last_action == 0:
            action_label = f"last: slow ({snapshot.slow_rate:.1f})"
        elif snapshot.last_action == 1:
            action_label = f"last: fast ({snapshot.fast_rate:.1f})"

        status_y = gauge.bottom + max(18, height // 10)
        self._draw_text(surface, f"level {snapshot.state}/{snapshot.buffer_size - 1}", (gauge.left, status_y), max(14, height // 16), TEXT_COLOR)
        self._draw_text(surface, action_label, (gauge.left + gauge.width // 3, status_y), max(14, height // 16), action_color)
        self._draw_text(surface, f"out {snapshot.out_rate:.1f}", (gauge.left + (gauge.width * 2) // 3, status_y), max(14, height // 16), MUTED_TEXT_COLOR)

        pulse_radius = max(5, height // 26) + (snapshot.step_count % 2) * max(1, height // 90)
        pulse_center = (gauge.right - max(18, width // 24), panel.top + max(26, height // 9))
        pygame.draw.circle(surface, action_color, pulse_center, pulse_radius)

    def _draw_vertical_marker(
        self,
        surface: Any,
        gauge: Any,
        state: int,
        buffer_size: int,
        color: RGBColor,
        label: str,
    ) -> None:
        import pygame

        x = gauge.left + int((state + 0.5) * gauge.width / buffer_size)
        pygame.draw.line(surface, color, (x, gauge.top - gauge.height // 2), (x, gauge.bottom + gauge.height // 2), width=max(2, gauge.height // 12))
        self._draw_text(surface, label, (x - gauge.height, gauge.top - gauge.height), max(10, gauge.height // 3), color)

    def _draw_text(self, surface: Any, text: str, pos: Position, size: int, color: RGBColor) -> None:
        import pygame

        pygame.font.init()
        font = pygame.font.Font(None, int(size))
        rendered = font.render(text, True, color)
        surface.blit(rendered, pos)


class _MediaStreamingSnapshot:
    def __init__(
        self,
        *,
        state: int,
        start_state: int,
        buffer_size: int,
        slow_rate: float,
        fast_rate: float,
        out_rate: float,
        last_action: int | None,
        step_count: int,
    ) -> None:
        self.state = state
        self.start_state = start_state
        self.buffer_size = buffer_size
        self.slow_rate = slow_rate
        self.fast_rate = fast_rate
        self.out_rate = out_rate
        self.last_action = last_action
        self.step_count = step_count


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


__all__ = ["MediaStreamingRenderer", "RGBColor", "validate_renderer_options"]
