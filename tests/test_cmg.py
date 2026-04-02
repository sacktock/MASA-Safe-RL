import pytest

from masa.common.constraints.multi_agent.cmg import Budget, ConstrainedMarkovGame, ConstrainedMarkovGameEnv
from masa.common.labelled_pz_env import LabelledParallelEnv
from masa.envs.multiagent.matrix.chicken import Actions, ChickenMatrix, cost_fn, label_fn


def test_budget_normalizes_agents():
    budget = Budget(amount=3, agents=("player_0", "player_0", "player_1"), name="shared")

    assert budget.amount == 3.0
    assert budget.agents == ("player_0", "player_1")


def test_constrained_markov_game_rejects_unknown_budget_agents():
    with pytest.raises(ValueError, match="possible_agents"):
        ConstrainedMarkovGame(
            possible_agents=("player_0", "player_1"),
            budgets=[Budget(amount=1.0, agents=("player_2",), name="invalid")],
            cost_fn=cost_fn,
        )


def test_constrained_markov_game_env_tracks_overlapping_budgets():
    env = ConstrainedMarkovGameEnv(
        LabelledParallelEnv(ChickenMatrix(max_moves=1), label_fn),
        budgets=[
            Budget(amount=1.0, agents=("player_0",), name="solo"),
            Budget(amount=1.5, agents=("player_0", "player_1"), name="shared"),
        ],
        cost_fn=cost_fn,
    )

    obs, infos = env.reset(seed=0)

    assert set(obs) == {"player_0", "player_1"}
    assert infos["player_0"]["labels"] == set()
    assert infos["player_1"]["labels"] == set()
    assert env.constraint_type == "cmg"
    assert env.constraint_step_metrics()["shared_cum_cost"] == 0.0

    _, _, _, truncations, infos = env.step(
        {"player_0": Actions.Straight, "player_1": Actions.Straight}
    )

    assert truncations["player_0"] is True
    assert truncations["player_1"] is True
    assert "unsafe" in infos["player_0"]["labels"]
    assert "unsafe" in infos["player_1"]["labels"]

    step_metrics = env.constraint_step_metrics()
    episode_metrics = env.constraint_episode_metrics()

    assert step_metrics["player_0_cost"] == pytest.approx(1.0)
    assert step_metrics["player_1_cost"] == pytest.approx(1.0)
    assert step_metrics["solo_cost"] == pytest.approx(1.0)
    assert step_metrics["shared_cost"] == pytest.approx(2.0)
    assert step_metrics["solo_satisfied"] == pytest.approx(1.0)
    assert step_metrics["shared_satisfied"] == pytest.approx(0.0)
    assert step_metrics["satisfied"] == pytest.approx(0.0)
    assert episode_metrics["player_0_cum_cost"] == pytest.approx(1.0)
    assert episode_metrics["player_1_cum_cost"] == pytest.approx(1.0)
    assert episode_metrics["solo_cum_cost"] == pytest.approx(1.0)
    assert episode_metrics["shared_cum_cost"] == pytest.approx(2.0)
    assert episode_metrics["solo_satisfied"] == pytest.approx(1.0)
    assert episode_metrics["shared_satisfied"] == pytest.approx(0.0)
    assert episode_metrics["satisfied"] == pytest.approx(0.0)
