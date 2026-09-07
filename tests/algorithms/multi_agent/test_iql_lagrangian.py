from __future__ import annotations

import numpy as np
import pytest

from masa.algorithms.multi_agent import IQLLagrangian


def test_dual_uses_episode_sum_and_updates_once_per_budget(model_factory, toy_env_factory):
    env = toy_env_factory(horizon=2)
    model = model_factory(
        IQLLagrangian,
        env=env,
        cost_lambda={"shared": 0.0},
        lambda_lr=0.1,
    )
    row = model.train(1)[0]

    assert row["shared_cum_cost"] == 4.0
    assert row["lambda/shared"] == 0.0
    assert row["lambda_next/shared"] == pytest.approx(0.25)
    assert model.lambdas["shared"] == pytest.approx(0.1 * (4.0 - 1.5))


def test_dual_batch_persists_across_train_calls_and_uses_mean(model_factory):
    model = model_factory(
        IQLLagrangian,
        cost_lambda={"shared": 0.0},
        lambda_lr=0.1,
        dual_update_every=2,
    )

    model.train(1)
    assert model.lambdas["shared"] == 0.0

    model.env.costs = (0.0, 0.0)
    model.train(1)
    assert model.lambdas["shared"] == 0.0  # mean cost 1.0 < budget 1.5

    model.env.costs = (1.0, 1.0)
    model.train(2)
    assert model.lambdas["shared"] == pytest.approx(0.05)


def test_dual_projects_to_zero_and_honours_cap(model_factory):
    model = model_factory(
        IQLLagrangian,
        cost_lambda={"shared": 0.02},
        lambda_lr=0.1,
        lambda_max=0.03,
    )
    model.train(1)
    assert model.lambdas["shared"] == 0.03

    model.env.costs = (0.0, 0.0)
    model.train(1)
    assert model.lambdas["shared"] == 0.0


def test_evaluation_freezes_dual_state(model_factory):
    model = model_factory(
        IQLLagrangian,
        cost_lambda={"shared": 0.0},
        dual_update_every=2,
    )
    model.train(1)

    lambdas = model.lambdas.copy()
    sums = model._dual_cost_sums.copy()
    count = model._dual_episode_count
    q_tables = {agent: table.copy() for agent, table in model.q_tables.items()}

    model.evaluate(3, deterministic=False)

    assert model.lambdas == lambdas
    assert model._dual_cost_sums == sums
    assert model._dual_episode_count == count
    for agent, table in model.q_tables.items():
        np.testing.assert_array_equal(table, q_tables[agent])


def test_lagrangian_defaults_gamma_to_one(model_factory):
    model = model_factory(
        IQLLagrangian,
        cost_lambda={"shared": 0.0},
        ql_kwargs={},
    )
    assert all(learner.gamma == 1.0 for learner in model.learners.values())


def test_lagrangian_validation(model_factory):
    with pytest.raises(ValueError, match="gamma"):
        model_factory(
            IQLLagrangian,
            cost_lambda={"shared": 0.0},
            ql_kwargs={"gamma": 0.9},
        )
    with pytest.raises(ValueError, match="lambda_lr"):
        model_factory(
            IQLLagrangian,
            cost_lambda={"shared": 0.0},
            lambda_lr=0.0,
        )
    with pytest.raises(ValueError, match="dual_update_every"):
        model_factory(
            IQLLagrangian,
            cost_lambda={"shared": 0.0},
            dual_update_every=0,
        )
    with pytest.raises(ValueError, match="lambda_max"):
        model_factory(
            IQLLagrangian,
            cost_lambda={"shared": 0.1},
            lambda_max=0.0,
        )
