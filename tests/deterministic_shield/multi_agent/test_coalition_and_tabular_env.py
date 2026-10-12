from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo import ParallelEnv

from masa.common.multi_agent import Coalition, LabelledParallelEnv
from masa.deterministic_shield import CoalitionLTLShield
from masa.envs.multiagent import TabularParallelEnv

from conftest import TinyParallelEnv, bad_dfa


def test_coalition_is_canonicalized_by_environment_order():
    coalition = Coalition(("b", "a"), name="team")
    assert coalition.agents == ("a", "b")
    assert coalition == Coalition(("a", "b"), name="team")
    assert coalition == Coalition(("a", "b"), name="other display name")
    assert coalition.ordered(("a", "b", "c")) == ("a", "b")
    assert "a" in coalition
    assert coalition.name == "team"


@pytest.mark.parametrize(
    "agents,error",
    [
        ((), "at least one"),
        (("a", "a"), "unique"),
        (("",), "non-empty strings"),
    ],
)
def test_coalition_validation(agents, error):
    with pytest.raises((TypeError, ValueError), match=error):
        Coalition(agents)


def test_coalition_rejects_unknown_agent_and_bare_string():
    with pytest.raises(ValueError, match="unknown"):
        Coalition(("missing",)).ordered(("a", "b"))
    with pytest.raises(TypeError, match="sequence"):
        Coalition("agent")


def test_joint_action_codec_uses_possible_agent_order():
    transitions = {
        (0, (a, b, c)): (0,)
        for a in range(2)
        for b in range(3)
        for c in range(2)
    }
    env = TinyParallelEnv(
        agents=("a", "b", "c"),
        n_states=1,
        action_sizes={"a": 2, "b": 3, "c": 2},
        transitions=transitions,
        labels=lambda state: set(),
    )
    assert env.n_joint_actions == 12
    assert env.encode_joint_action({"c": 1, "a": 1, "b": 2}) == 11
    assert env.decode_joint_action(11) == {"a": 1, "b": 2, "c": 1}
    assert env.encode_joint_action((1, 0, 1)) == 7


class DenseOneAgent(TabularParallelEnv):
    def __init__(self, matrix):
        super().__init__()
        self.possible_agents = ["a"]
        self.agents = ["a"]
        self._n_states = 2
        self._state = 0
        self._transition_matrix = np.asarray(matrix)
        self._action_space = spaces.Discrete(2)
        self._observation_space = spaces.Discrete(2)

    def action_space(self, agent):
        return self._action_space

    def observation_space(self, agent):
        return self._observation_space


def test_dense_support_includes_every_positive_probability():
    matrix = np.zeros((2, 2, 2))
    matrix[0, 0, 0] = 1.0
    matrix[0, 0, 1] = 1.0 - 1e-14
    matrix[1, 0, 1] = 1e-14
    matrix[1, 1, :] = 1.0
    env = DenseOneAgent(matrix)
    assert env.successors(0, (0,)) == (0,)
    assert env.successors(0, (1,)) == (0, 1)
    assert env.observations_from_state(1) == {"a": 1}


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, 0.2])
def test_dense_support_rejects_invalid_probabilities(value):
    matrix = np.zeros((2, 2, 2))
    matrix[0, 0, 0] = value
    matrix[0, 0, 1] = 1.0
    matrix[1, 1, :] = 1.0
    env = DenseOneAgent(matrix)
    with pytest.raises(ValueError):
        env.successors(0, (0,))


def test_sparse_support_rejects_missing_joint_action():
    env = TinyParallelEnv(
        agents=("a",),
        n_states=1,
        action_sizes={"a": 2},
        transitions={(0, (0,)): (0,)},
        labels=lambda state: set(),
    )
    with pytest.raises(ValueError, match="sum to 1|wrong shape"):
        env.successors(0, (1,))


def test_state_dependent_legal_actions_are_validated():
    transitions = {
        (0, (0, 0)): (0,),
        (0, (2, 0)): (0,),
        (1, (0, 0)): (1,),
        (1, (1, 0)): (1,),
    }
    env = TinyParallelEnv(
        agents=("a", "b"),
        n_states=2,
        action_sizes={"a": 3, "b": 1},
        transitions=transitions,
        labels=lambda state: set(),
        legal_actions_fn=lambda state, agent: (
            (0, 2) if state == 0 and agent == "a" else (0, 1)
            if agent == "a"
            else (0,)
        ),
    )
    assert env.get_legal_actions(0, "a") == (0, 2)
    assert env.successors(0, (2, 0)) == (0,)
    with pytest.raises(ValueError, match="not legal"):
        env.successors(0, (1, 0))


class PlainParallel(ParallelEnv):
    possible_agents = ["a"]
    agents = ["a"]

    def action_space(self, agent):
        return spaces.Discrete(1)

    def observation_space(self, agent):
        return spaces.Discrete(1)


def test_shield_requires_labelled_tabular_parallel_env_directly():
    plain = PlainParallel()
    labelled = LabelledParallelEnv(plain, lambda obs: set())
    with pytest.raises(TypeError, match="TabularParallelEnv"):
        CoalitionLTLShield(labelled, coalition=("a",), dfa=bad_dfa())
