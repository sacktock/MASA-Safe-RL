"""Small modeled environments shared by the shielding tests."""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.common.labelled_env import LabelledEnv
from masa.common.ltl import Atom, DFA


class TinyMDP(gym.Env):
    def __init__(self, model, labels, *, sparse=False, ending=None):
        self.model = np.array(model, dtype=np.float64, copy=True)
        self.actual_model = self.model.copy()
        n_states, _, n_actions = self.model.shape
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(n_actions)
        self.labels = labels
        self.has_successor_states_dict = sparse
        self.has_transition_matrix = not sparse
        self.ending = ending
        self.executed = []
        self.fail_step = False

    def get_transition_matrix(self):
        return self.model

    def get_successor_states_dict(self):
        successors, probabilities = {}, {}
        for s in range(self.observation_space.n):
            ids = np.flatnonzero((self.model[:, s, :] > 0).any(axis=1))
            successors[s] = ids.tolist()
            for a in range(self.action_space.n):
                probabilities[s, a] = self.model[ids, s, a].copy()
        return successors, probabilities

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = int((options or {}).get("state", 0))
        self.executed.clear()
        return self.state, {"base_info": "preserved"}

    def step(self, action):
        if self.fail_step:
            raise RuntimeError("simulated environment failure")
        self.executed.append(int(action))
        self.state = int(self.np_random.choice(
            self.observation_space.n, p=self.actual_model[:, self.state, action]
        ))
        return (
            self.state, 0.25, self.ending == "terminated",
            self.ending == "truncated", {"base_info": "preserved"},
        )


@pytest.fixture
def shield_class(request):
    return request.param


@pytest.fixture
def make_shield():
    def make(cls, *, model=None, labels=None, dfa=None, sparse=False,
             ending=None, **shield_options):
        if model is None:
            # State 1 is a delayed trap; state 2 is immediately bad.
            model = np.zeros((3, 3, 4))
            model[0, 0, [0, 3]] = 1
            model[1, 0, 1] = 1
            model[2, 0, 2] = 1
            model[2, 1, :] = 1
            model[2, 2, :] = 1
            labels = [set(), set(), {"bad"}]
        if labels is None:
            labels = [set() for _ in range(model.shape[0])]
        if dfa is None:
            dfa = DFA([0, 1], 0, [1])
            dfa.add_edge(0, 1, Atom("bad"))
        base = TinyMDP(model, labels, sparse=sparse, ending=ending)
        monitor = LTLSafetyEnv(
            LabelledEnv(base, lambda s: base.labels[s]),
            dfa=dfa, obs_type="discrete",
        )
        return cls(monitor, **shield_options)
    return make
