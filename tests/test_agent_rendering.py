import numpy as np
import pytest

from masa.common.rendering.agent import SHIRT_COLOURS, draw_agent, shirt_colour


def test_player_shirt_colours_start_blue_then_red() -> None:
    assert shirt_colour(1) == (55, 115, 206)
    assert shirt_colour(2) == (216, 93, 83)
    assert shirt_colour(3) == SHIRT_COLOURS[2]


def test_player_shirt_colours_cycle_for_arbitrary_agent_ids() -> None:
    for agent_id in range(1, len(SHIRT_COLOURS) * 3 + 1):
        assert shirt_colour(agent_id) == SHIRT_COLOURS[(agent_id - 1) % len(SHIRT_COLOURS)]


def test_agent_ids_are_one_indexed() -> None:
    with pytest.raises(ValueError, match="agent_id must be >= 1"):
        shirt_colour(0)

    with pytest.raises(ValueError, match="agent_id must be >= 1"):
        draw_agent(None, (0, 0), 32, agent_id=0)


def test_cell_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="cell_size must be positive"):
        draw_agent(None, (0, 0), 0)


def test_draw_agent_supports_numpy_rgb_frames() -> None:
    pytest.importorskip("pygame")

    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    draw_agent(frame, (48, 48), 64, agent_id=2)

    red_shirt = np.asarray(shirt_colour(2), dtype=np.uint8)
    assert np.any(frame)
    assert np.any(np.all(frame == red_shirt, axis=2))
