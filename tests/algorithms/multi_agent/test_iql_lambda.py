from __future__ import annotations

import numpy as np
import pytest

from masa.algorithms.multi_agent import IQL, IQLLambda


def test_zero_fixed_penalty_matches_plain_iql(model_factory):
    plain = model_factory(IQL)
    fixed = model_factory(IQLLambda, cost_lambda={"shared": 0.0})

    plain.train(5)
    fixed.train(5)

    assert plain.env.actions == fixed.env.actions
    for agent in plain.agents:
        np.testing.assert_array_equal(plain.q_tables[agent], fixed.q_tables[agent])


def test_shared_penalty_uses_aggregate_budget_cost_once(model_factory):
    model = model_factory(
        IQLLambda,
        cost_lambda={"shared": 3.0},
        ql_kwargs={"alpha": 1.0},
    )
    row = model.train(1)[0]

    assert row["return/player_0"] == 1.0
    assert row["shaped_return/player_0"] == -5.0  # 1 - 3 * (1 + 1)
    for agent in model.agents:
        action = model.env.actions[0][agent]
        assert model.q_tables[agent][0, action] == -5.0
    assert model.lambdas == {"shared": 3.0}


def test_overlapping_selected_budgets_add_only_for_members(model_factory):
    model = model_factory(
        IQLLambda,
        cost_lambda={"local_0": 2.0, "shared": 3.0},
    )
    metrics = model.env.constraint_step_metrics()

    assert model._penalty("player_0", metrics) == 8.0
    assert model._penalty("player_1", metrics) == 6.0


def test_scalar_selects_all_budgets(model_factory):
    model = model_factory(IQLLambda, cost_lambda=0.5)
    assert model.lambdas == {"local_0": 0.5, "local_1": 0.5, "shared": 0.5}


def test_fixed_multiplier_validation(model_factory):
    with pytest.raises(ValueError, match="Unknown"):
        model_factory(IQLLambda, cost_lambda={"typo": 1.0})
    with pytest.raises(ValueError, match="nonnegative"):
        model_factory(IQLLambda, cost_lambda=-1.0)
    with pytest.raises(ValueError, match="Select"):
        model_factory(IQLLambda, cost_lambda={})
