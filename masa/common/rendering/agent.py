from __future__ import annotations

import os
from typing import Any

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

RGBColor = tuple[int, int, int]
Position = tuple[int, int]

# Player shirts cycle through this palette for agent ids greater than the
# palette length. P1 is deliberately blue and P2 red for quick recognition.
SHIRT_COLOURS: tuple[RGBColor, ...] = (
    (55, 115, 206),   # P1 - blue
    (216, 93, 83),    # P2 - red
    (78, 166, 129),   # P3 - green
    (151, 101, 201),  # P4 - purple
    (230, 153, 70),   # P5 - orange
    (59, 158, 174),   # P6 - teal
)

_OUTLINE: RGBColor = (45, 48, 55)
_SKIN: RGBColor = (242, 188, 143)
_HAIR: RGBColor = (88, 54, 32)
_TEXT: RGBColor = (255, 255, 255)
_SHADOW: RGBColor = (130, 126, 118)


def shirt_colour(agent_id: int) -> RGBColor:
    """Return the cyclic shirt colour for a one-indexed player id."""
    if agent_id < 1:
        raise ValueError("agent_id must be >= 1")
    return SHIRT_COLOURS[(agent_id - 1) % len(SHIRT_COLOURS)]


def draw_agent(
    target: Any,
    center: Position,
    cell_size: int,
    *,
    agent_id: int = 1,
    step_count: int = 0,
) -> None:
    """Draw a compact, sprite-like player avatar onto a render target.

    ``target`` may be either a ``pygame.Surface`` or an ``H x W x 3`` NumPy
    RGB array. Agent ids are one-indexed: P1 is blue, P2 is red, and later
    players cycle through :data:`SHIRT_COLOURS`. The shirt label is always
    rendered as ``P{agent_id}`` and is automatically shrunk when necessary.
    """
    if agent_id < 1:
        raise ValueError("agent_id must be >= 1")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    import pygame

    frame: np.ndarray | None = None
    if isinstance(target, np.ndarray):
        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("NumPy render targets must have shape (H, W, 3)")
        frame = target
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    else:
        surface = target

    cx, cy = int(center[0]), int(center[1])
    bob = -max(1, cell_size // 40) if step_count % 2 else 0
    cy += bob

    shirt = shirt_colour(agent_id)

    # Ground shadow.
    shadow = pygame.Rect(
        cx - int(cell_size * 0.30),
        cy + int(cell_size * 0.27),
        int(cell_size * 0.60),
        max(3, int(cell_size * 0.08)),
    )
    pygame.draw.ellipse(surface, _SHADOW, shadow)

    # Large torso so the player label remains readable after downsampling.
    torso_w = max(1, int(cell_size * 0.58))
    torso_h = max(1, int(cell_size * 0.40))
    torso = pygame.Rect(
        cx - torso_w // 2,
        cy - int(cell_size * 0.01),
        torso_w,
        torso_h,
    )
    corner = max(3, cell_size // 15)
    stroke = max(2, cell_size // 35)

    pygame.draw.rect(surface, _OUTLINE, torso, border_radius=corner)
    inner = torso.inflate(-stroke * 2, -stroke * 2)
    pygame.draw.rect(
        surface,
        shirt,
        inner,
        border_radius=max(2, corner - stroke),
    )

    # Head and simple hair silhouette. Chunky shapes survive small grid cells
    # more reliably than detailed facial features.
    head_r = max(2, int(cell_size * 0.22))
    head_center = (cx, cy - int(cell_size * 0.22))
    pygame.draw.circle(surface, _OUTLINE, head_center, head_r + stroke)
    pygame.draw.circle(surface, _SKIN, head_center, head_r)

    hair_rect = pygame.Rect(
        head_center[0] - head_r,
        head_center[1] - head_r,
        head_r * 2,
        max(1, int(head_r * 1.15)),
    )
    pygame.draw.arc(
        surface,
        _HAIR,
        hair_rect,
        0,
        3.14159,
        width=max(4, int(head_r * 0.55)),
    )

    fringe_y = head_center[1] - int(head_r * 0.55)
    for dx in (-0.45, 0.0, 0.45):
        pygame.draw.circle(
            surface,
            _HAIR,
            (head_center[0] + int(head_r * dx), fringe_y),
            max(3, int(head_r * 0.30)),
        )

    eye_r = max(2, int(cell_size * 0.025))
    eye_dx = int(head_r * 0.38)
    eye_y = head_center[1] + int(head_r * 0.08)
    pygame.draw.circle(surface, _OUTLINE, (cx - eye_dx, eye_y), eye_r)
    pygame.draw.circle(surface, _OUTLINE, (cx + eye_dx, eye_y), eye_r)

    if not pygame.font.get_init():
        pygame.font.init()

    label = f"P{agent_id}"
    font_size = max(18, int(cell_size * 0.34))
    max_text_width = max(1, int(inner.width * 0.88))
    max_text_height = max(1, int(inner.height * 0.72))

    # P1/P2 stay large; wider labels such as P12 are scaled down to fit.
    while True:
        font = pygame.font.Font(None, font_size)
        font.set_bold(True)
        text = font.render(label, True, _TEXT)
        if (
            (text.get_width() <= max_text_width and text.get_height() <= max_text_height)
            or font_size <= 8
        ):
            break
        font_size -= 1

    text_rect = text.get_rect(
        center=(cx, torso.centery + int(cell_size * 0.015))
    )
    surface.blit(text, text_rect)

    if frame is not None:
        rendered = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        frame[...] = rendered


__all__ = ["SHIRT_COLOURS", "draw_agent", "shirt_colour"]
