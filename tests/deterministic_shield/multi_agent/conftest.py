from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import product

import numpy as np
import pytest
from gymnasium import spaces

from masa.common.ltl import Atom, DFA
from masa.common.multi_agent import LabelledParallelEnv
from masa.envs.multiagent import TabularParallelEnv


class TinyParallelEnv(TabularParallelEnv):
    metadata = {"name": "tiny_tabular_parallel_v0", "is_parallelizable": True}

    def __init__(
        self,
        *,
        agents=("a", "b"),
        n_states=2,
        action_sizes=None,
        transitions: Mapping[tuple[int, tuple[int, ...]], tuple[int, ...]],
        runtime_transitions=None,
        labels: Callable[[int], set[str]] | None = None,
        legal_actions_fn=None,
        finish: str | None = None,
    ):
        super().__init__()
        self.possible_agents = list(agents)
        self.agents = list(agents)
        self._n_states = int(n_states)
        self._action_sizes = action_sizes or {agent: 2 for agent in agents}
        self.action_spaces = {
            agent: spaces.Discrete(self._action_sizes[agent])
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: spaces.Discrete(self.n_states)
            for agent in self.possible_agents
        }
        self.transitions = {
            (int(state), tuple(action)): tuple(successors)
            for (state, action), successors in transitions.items()
        }
        self.runtime_transitions = (
            self.transitions
            if runtime_transitions is None
            else {
                (int(state), tuple(action)): tuple(successors)
                for (state, action), successors in runtime_transitions.items()
            }
        )
        self.labels = labels or (
            lambda state: {"bad"} if state == 1 else set()
        )
        self._legal_actions_fn = legal_actions_fn
        self.finish = finish
        self._state = 0
        self.step_count = 0
        self.successor_choice = 0
        self.last_actions = None
        self._install_sparse_model()

    def _install_sparse_model(self):
        successor_states = {}
        transition_probs = {}
        sizes = tuple(self._action_sizes[agent] for agent in self.possible_agents)
        for state in range(self.n_states):
            row_successors = sorted(
                {
                    successor
                    for action in product(*(range(size) for size in sizes))
                    for successor in self.transitions.get((state, tuple(action)), ())
                }
            )
            successor_states[state] = row_successors
            index = {successor: i for i, successor in enumerate(row_successors)}
            for action in product(*(range(size) for size in sizes)):
                action = tuple(action)
                support = self.transitions.get((state, action))
                if support is None:
                    continue
                probs = np.zeros(len(row_successors), dtype=np.float64)
                for successor in support:
                    probs[index[successor]] += 1.0 / len(support)
                transition_probs[(state, self.encode_joint_action(action))] = probs
        self._successor_states = successor_states
        self._transition_probs = transition_probs

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def legal_actions(self, state: int, agent: str):
        if self._legal_actions_fn is None:
            return super().legal_actions(state, agent)
        return self._legal_actions_fn(state, agent)

    def reset(self, seed=None, options=None):
        del seed
        self.agents = list(self.possible_agents)
        self._state = int((options or {}).get("state", 0))
        self.step_count = 0
        self.last_actions = None
        return self.observations_from_state(self._state), {
            agent: {"base": True} for agent in self.possible_agents
        }

    def step(self, actions):
        joint = tuple(int(actions[agent]) for agent in self.possible_agents)
        successors = self.runtime_transitions[(self.get_state_id(), joint)]
        self.last_actions = dict(actions)
        self._state = int(successors[self.successor_choice % len(successors)])
        self.step_count += 1
        observations = self.observations_from_state(self._state)
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {
            agent: self.finish == "terminated" for agent in self.possible_agents
        }
        truncations = {
            agent: self.finish == "truncated" for agent in self.possible_agents
        }
        infos = {agent: {"base": True} for agent in self.possible_agents}
        if self.finish is not None:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self):
        return self.get_state_id()


def bad_dfa(label="bad"):
    dfa = DFA([0, 1], initial=0, accepting=[1])
    dfa.add_edge(0, 1, Atom(label))
    return dfa


def labelled(env: TinyParallelEnv):
    return LabelledParallelEnv(env, env.labels)


@pytest.fixture
def all_except_11_env():
    transitions = {}
    for state in range(2):
        for a in range(2):
            for b in range(2):
                successor = 1 if state == 1 or (a == 1 and b == 1) else 0
                transitions[(state, (a, b))] = (successor,)
    return TinyParallelEnv(transitions=transitions)
