from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from masa.algorithms.tabular.multi_agent.iql import IQL, _AgentSpaces, _ExternalQL


def test_external_adapter_reuses_native_q_update_and_schedule():
    learner = _ExternalQL(
        _AgentSpaces(3, spaces.Discrete(2)),
        seed=0,
        alpha=0.5,
        gamma=1.0,
        exploration="epsilon_greedy",
        initial_epsilon=1.0,
        final_epsilon=0.0,
        epsilon_decay_frames=10,
    )
    learner.Q[1] = [4.0, 6.0]
    learner.observe(0, 1, -2.0, 1, False)

    assert learner.Q[0, 1] == pytest.approx(2.0)
    assert learner._step == 1
    assert learner._epsilon == pytest.approx(0.9)
    assert not learner.buffer

    learner.Q[2] = 1000.0
    learner.observe(0, 0, -2.0, 2, True)
    assert learner.Q[0, 0] == pytest.approx(-1.0)


def test_agent_spaces_cannot_advance_shared_game():
    view = _AgentSpaces(2, spaces.Discrete(2))
    with pytest.raises(RuntimeError):
        view.reset()
    with pytest.raises(RuntimeError):
        view.step(0)


def test_one_parallel_step_per_round_and_separate_q_tables(model_factory, toy_env_factory):
    env = toy_env_factory(horizon=3)
    model = model_factory(IQL, env=env)
    rows = model.train(2)

    assert env.steps == 6
    assert rows[-1]["return/player_0"] == 3.0
    assert model.learners["player_0"]._step == 6
    assert model.learners["player_1"]._step == 6
    assert model.q_tables["player_0"] is not model.q_tables["player_1"]


def test_joint_action_is_complete_before_any_update(model_factory, monkeypatch):
    model = model_factory(IQL)
    events = []

    for agent, learner in model.learners.items():
        original_act = learner.act
        original_observe = learner.observe

        def record_act(*args, _agent=agent, _fn=original_act, **kwargs):
            events.append(("act", _agent))
            return _fn(*args, **kwargs)

        def record_update(*args, _agent=agent, _fn=original_observe, **kwargs):
            events.append(("update", _agent))
            return _fn(*args, **kwargs)

        monkeypatch.setattr(learner, "act", record_act)
        monkeypatch.setattr(learner, "observe", record_update)

    model.train(1)
    assert [kind for kind, _ in events] == ["act", "act", "update", "update"]


def test_evaluation_does_not_change_learning_state(model_factory):
    model = model_factory(IQL)
    model.train(2)
    snapshot = {
        agent: (learner.Q.copy(), learner._step, learner._epsilon)
        for agent, learner in model.learners.items()
    }

    model.evaluate(3, deterministic=False)
    model.evaluate(3, deterministic=True)

    for agent, learner in model.learners.items():
        q_table, step, epsilon = snapshot[agent]
        np.testing.assert_array_equal(learner.Q, q_table)
        assert learner._step == step
        assert learner._epsilon == epsilon
    assert model.episodes == 2


def test_evaluation_does_not_change_subsequent_training(model_factory):
    with_eval = model_factory(IQL)
    without_eval = model_factory(IQL)

    with_eval.train(2)
    without_eval.train(2)
    with_eval.evaluate(3, deterministic=False)
    with_eval.train(2)
    without_eval.train(2)

    for agent in with_eval.agents:
        np.testing.assert_array_equal(
            with_eval.q_tables[agent],
            without_eval.q_tables[agent],
        )


def test_discrete_observation_size_is_inferred(model_factory):
    model = model_factory(IQL)
    assert model._n_states == {"player_0": 1, "player_1": 1}
    assert model._state("player_0", 0, 0) == 0


def test_time_is_part_of_the_table_state(toy_env_factory):
    env = toy_env_factory(horizon=3)
    model = IQL(
        env,
        horizon=3,
        n_states=4,
        encode=lambda observation: int(observation),
        ql_kwargs={"gamma": 1.0},
    )
    assert model._state("player_0", 0, 0) == 0
    assert model._state("player_0", 0, 1) == 4
    assert model._state("player_0", 3, 2) == 11


def test_per_agent_state_configuration_is_supported(toy_env_factory):
    env = toy_env_factory(horizon=2)
    model = IQL(
        env,
        horizon=2,
        n_states={"player_0": 2, "player_1": 3},
        encode={"player_0": int, "player_1": int},
    )
    assert model.q_tables["player_0"].shape == (4, 2)
    assert model.q_tables["player_1"].shape == (6, 2)


def test_invalid_state_configuration_and_horizon_fail_loudly(toy_env_factory):
    env = toy_env_factory(horizon=2)
    with pytest.raises(ValueError, match="horizon"):
        IQL(env, horizon=0)
    with pytest.raises(ValueError, match="exactly"):
        IQL(env, horizon=2, n_states={"player_0": 1})

    model = IQL(env, horizon=1)
    with pytest.raises(ValueError, match="horizon"):
        model.train(1)
