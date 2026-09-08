"""Integration checks against MASA's repeated Chicken ParallelEnv."""
from __future__ import annotations

import numpy as np
import pytest

from masa.common.multi_agent import Coalition
from masa.deterministic_shield import CoalitionLTLShield
from masa.envs.multiagent import TabularParallelEnv
from masa.envs.multiagent.matrix.chicken import Actions, ChickenMatrix
from masa.examples.det_shield_coalitional_chicken_safety_game import (
    make_labelled_chicken_env,
    make_never_crash_dfa,
)


def make_shield(*, coalition, mode="preemptive", execution="centralised"):
    return CoalitionLTLShield(
        make_labelled_chicken_env(max_moves=2),
        coalition=Coalition(tuple(coalition)),
        dfa=make_never_crash_dfa(),
        mode=mode,
        execution=execution,
    )


def test_chicken_exposes_its_exact_tabular_parallel_model():
    env = ChickenMatrix(max_moves=2)
    assert isinstance(env, TabularParallelEnv)
    assert env.n_states == 5
    assert env.n_joint_actions == 4
    assert env.successors(0, (Actions.Swerve, Actions.Straight)) == (2,)
    assert env.successors(4, (Actions.Straight, Actions.Straight)) == (4,)

    observations = env.observations_from_state(3)
    expected = np.array([0, 1, 1, 0, 0], dtype=np.uint8)
    np.testing.assert_array_equal(observations["player_0"], expected)
    np.testing.assert_array_equal(observations["player_1"], expected)


def test_singleton_coalition_must_swerve_against_unrestricted_opponent():
    env = make_shield(coalition=("player_0",))
    env.reset(seed=0)
    assert env.tabular_state == 0
    assert env.labels_for_state(4) >= {"crash", "unsafe"}
    np.testing.assert_array_equal(env.coalition_action_mask(), [True, False])

    with pytest.raises(ValueError, match="unsafe"):
        env.step(
            {
                "player_0": Actions.Straight,
                "player_1": Actions.Swerve,
            }
        )

    _, _, _, _, infos = env.step(
        {
            "player_0": Actions.Swerve,
            "player_1": Actions.Straight,
        }
    )
    assert "crash" not in infos["player_0"]["labels"]
    env.close()


def test_decentralised_team_replacement_respects_cartesian_masks():
    env = make_shield(
        coalition=("player_0", "player_1"),
        mode="postposed",
        execution="decentralised",
    )
    env.reset(seed=0)
    np.testing.assert_array_equal(env.local_action_mask("player_0"), [True, True])
    np.testing.assert_array_equal(env.local_action_mask("player_1"), [True, False])

    _, _, _, _, infos = env.step(
        {
            "player_0": Actions.Straight,
            "player_1": Actions.Straight,
        }
    )
    assert infos["player_0"]["shield_executed_action"] == Actions.Straight
    assert infos["player_1"]["shield_executed_action"] == Actions.Swerve
    assert "crash" not in infos["player_0"]["labels"]
    env.close()


def test_chicken_state_snapshot_restores_tabular_state_id():
    env = ChickenMatrix(max_moves=3)
    env.reset(seed=0)
    env.step(
        {
            "player_0": Actions.Straight,
            "player_1": Actions.Swerve,
        }
    )
    snapshot = env.get_state()
    assert env.get_state_id() == 3

    env.reset(seed=1)
    assert env.get_state_id() == 0
    env.set_state(snapshot)
    assert env.get_state_id() == 3
    np.testing.assert_array_equal(
        env.state(), np.array([0, 1, 1, 0, 0], dtype=np.uint8)
    )
