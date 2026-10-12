from __future__ import annotations

import pytest

from masa.examples.cmg_iql import ALGORITHM_NAMES, run_variant


@pytest.mark.parametrize("algorithm", ALGORITHM_NAMES)
def test_chicken_cmg_example_smoke(algorithm):
    result = run_variant(
        algorithm,
        episodes=3,
        horizon=3,
        eval_episodes=2,
        seed=0,
        dual_update_every=2,
    )

    assert len(result["train"]) == 3
    assert len(result["eval"]) == 2
    assert set(result["q_tables"]) == {"player_0", "player_1"}
    assert all(row["steps"] == 3 for row in result["train"])
    assert all(row["shared_cum_cost"] >= 0.0 for row in result["train"])
