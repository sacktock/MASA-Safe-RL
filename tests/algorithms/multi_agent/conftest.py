"""Shared deterministic CMG fixtures for independent-learning tests."""
from __future__ import annotations

from collections.abc import Callable

import pytest
from gymnasium import spaces
from pettingzoo import ParallelEnv

from masa.common.constraints.multi_agent.cmg import Budget


class ToyCMG(ParallelEnv):
    """Two-agent deterministic CMG with configurable per-agent step costs."""

    metadata = {"name": "toy_cmg_v0"}
    constraint_type = "CMG"
    possible_agents = ("player_0", "player_1")

    def __init__(self, *, horizon: int = 1, costs: tuple[float, float] = (1.0, 1.0)):
        self.horizon = horizon
        self.costs = costs
        self.budgets = (
            Budget(1.0, ("player_0",), name="local_0"),
            Budget(1.0, ("player_1",), name="local_1"),
            Budget(1.5, self.possible_agents, name="shared"),
        )
        self.agents: list[str] = []
        self.steps = 0
        self.actions: list[dict[str, int]] = []
        self.round = 0
        self.shared_total = 0.0

    def observation_space(self, agent: str):
        return spaces.Discrete(1)

    def action_space(self, agent: str):
        return spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.round = 0
        self.shared_total = 0.0
        observations = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.steps += 1
        self.actions.append(dict(actions))
        self.round += 1
        self.shared_total += sum(self.costs)

        done = self.round >= self.horizon
        rewards = {agent: 1.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if done:
            self.agents = []
        observations = {agent: 0 for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def constraint_step_metrics(self):
        return {
            "local_0_cost": self.costs[0],
            "local_1_cost": self.costs[1],
            "shared_cost": sum(self.costs),
        }

    def constraint_episode_metrics(self):
        local_0 = self.round * self.costs[0]
        local_1 = self.round * self.costs[1]
        return {
            "player_0_cum_cost": local_0,
            "player_1_cum_cost": local_1,
            "local_0_cum_cost": local_0,
            "local_1_cum_cost": local_1,
            "shared_cum_cost": self.shared_total,
            "local_0_satisfied": float(local_0 <= 1.0),
            "local_1_satisfied": float(local_1 <= 1.0),
            "shared_satisfied": float(self.shared_total <= 1.5),
            "satisfied": float(
                local_0 <= 1.0 and local_1 <= 1.0 and self.shared_total <= 1.5
            ),
        }


@pytest.fixture
def toy_env_factory() -> Callable[..., ToyCMG]:
    return ToyCMG


@pytest.fixture
def model_factory(toy_env_factory):
    def make(model_class, *, env=None, ql_kwargs=None, **kwargs):
        env = toy_env_factory() if env is None else env
        config = {
            "exploration": "epsilon_greedy",
            **dict(ql_kwargs or {}),
        }
        return model_class(
            env,
            horizon=env.horizon,
            seed=7,
            ql_kwargs=config,
            **kwargs,
        )

    return make
