from __future__ import annotations

import numpy as np


def test_frozen_lake_is_registered_and_labels_default_map():
    import masa
    from gymnasium import spaces
    from masa.common.registry import ENV_REGISTRY
    from masa.envs.tabular.frozen_lake import FrozenLake, cost_fn, label_fn

    del masa

    assert ENV_REGISTRY.get("FrozenLake").__name__ == "FrozenLake"

    env = FrozenLake(is_slippery=False)
    assert isinstance(env.observation_space, spaces.Discrete)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.observation_space.n == 16
    assert env.action_space.n == 4
    assert env.label_fn(0) == {"start"}
    assert env.label_fn(1) == {"frozen"}
    assert env.label_fn(5) == {"hole"}
    assert env.label_fn(15) == {"goal"}
    assert label_fn(5) == {"hole"}
    assert cost_fn({"hole"}) == 1.0
    assert cost_fn({"frozen"}) == 0.0
    env.close()


def test_frozen_lake_transition_matrix_matches_deterministic_actions():
    from masa.envs.tabular.frozen_lake import FrozenLake

    env = FrozenLake(is_slippery=False)
    matrix = env.get_transition_matrix()

    assert env.has_transition_matrix is True
    assert env.has_successor_states_dict is False
    assert env.get_successor_states_dict() is None
    assert matrix.shape == (16, 16, 4)
    np.testing.assert_allclose(matrix.sum(axis=0), 1.0)

    assert matrix[0, 0, 0] == 1.0  # left into boundary
    assert matrix[4, 0, 1] == 1.0  # down
    assert matrix[1, 0, 2] == 1.0  # right
    assert matrix[0, 0, 3] == 1.0  # up into boundary
    env.close()


def test_frozen_lake_transition_matrix_aggregates_slippery_duplicates():
    from masa.envs.tabular.frozen_lake import FrozenLake

    env = FrozenLake(is_slippery=True, success_rate=1.0 / 3.0)
    matrix = env.get_transition_matrix()

    np.testing.assert_allclose(matrix.sum(axis=0), 1.0)
    np.testing.assert_allclose(matrix[0, 0, 0], 2.0 / 3.0)
    np.testing.assert_allclose(matrix[4, 0, 0], 1.0 / 3.0)
    np.testing.assert_allclose(matrix[0, 0, 3], 2.0 / 3.0)
    np.testing.assert_allclose(matrix[1, 0, 3], 1.0 / 3.0)
    env.close()


def test_frozen_lake_accepts_gymnasium_variants_and_custom_desc():
    from masa.envs.tabular.frozen_lake import FrozenLake

    large_env = FrozenLake(map_name="8x8", is_slippery=True, success_rate=0.9)
    assert large_env.observation_space.n == 64
    large_env.close()

    random_env = FrozenLake(map_name=None, is_slippery=False)
    assert random_env.observation_space.n == 64
    random_env.close()

    custom_env = FrozenLake(
        desc=["SF", "HG"],
        is_slippery=False,
        success_rate=0.8,
        reward_schedule=(2.0, -1.0, 0.0),
    )
    assert custom_env.observation_space.n == 4
    assert custom_env.label_fn(2) == {"hole"}
    assert custom_env.label_fn(3) == {"goal"}

    obs, _ = custom_env.reset(seed=0)
    assert obs == 0
    obs, reward, terminated, truncated, _ = custom_env.step(1)
    assert obs == 2
    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    custom_env.close()


def test_frozen_lake_make_env_uses_default_labels_and_costs():
    from masa.common.utils import make_env

    env = make_env(
        "FrozenLake",
        "PCTL",
        100,
        env_kwargs={"is_slippery": False},
        constraint_kwargs={"alpha": 0.01},
    )

    obs, info = env.reset(seed=0)
    assert obs == 0
    assert info["labels"] == {"start"}
    assert info["constraint"]["type"] == "PCTL"
    assert info["constraint"]["step"]["cost"] == 0.0

    env.close()


def test_frozen_lake_prob_shield_constructs_and_steps():
    from masa.common.utils import make_env
    from masa.prob_shield.prob_shield_wrapper_v1 import ProbShieldWrapperDisc

    env = make_env(
        "FrozenLake",
        "PCTL",
        100,
        env_kwargs={"is_slippery": False},
        constraint_kwargs={"alpha": 0.01},
    )
    shielded_env = ProbShieldWrapperDisc(
        env,
        init_safety_bound=1e-12,
        theta=1e-12,
        max_vi_steps=2_000,
        granularity=10,
    )

    obs, info = shielded_env.reset(seed=0)
    assert set(obs) == {"orig_obs", "safety_bound"}
    assert obs["orig_obs"] == 0
    assert info["labels"] == {"start"}

    action = np.zeros(2 + shielded_env.max_successors, dtype=np.int64)
    action[0] = 2
    action[1] = 1

    next_obs, reward, terminated, truncated, info = shielded_env.step(action)
    assert set(next_obs) == {"orig_obs", "safety_bound"}
    assert shielded_env._orig_obs_space.contains(next_obs["orig_obs"])
    assert reward in (0.0, 1.0)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "proj_penalty" in info
    assert "margin_penalty" in info

    shielded_env.close()
