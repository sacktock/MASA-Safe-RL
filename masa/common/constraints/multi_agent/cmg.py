"""
Overview
--------

Constraint monitors for labelled PettingZoo parallel environments.

This module provides a constrained Markov game monitor based on cumulative
cost budgets. Each agent receives its own label set through
``infos[agent]["labels"]`` and incurs a step cost via ``cost_fn(labels)``.
Budgets are then evaluated over subsets of agents:

.. math::

   C_t^{(i)} = \mathrm{cost}(L_t^{(i)}),

.. math::

   B_t^{(k)} = \sum_{i \in \mathcal{G}_k} C_t^{(i)},

where :math:`\mathcal{G}_k` is the subset of agents assigned to budget
:math:`k`. Budgets may overlap, so the same agent cost can contribute to
multiple budget totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

from pettingzoo import ParallelEnv

from masa.common.constraints.base import CostFn
from masa.common.dummy import cost_fn as dummy_cost_fn
from masa.common.labelled_pz_env import LabelledParallelEnv


@dataclass(frozen=True, slots=True)
class Budget:
    """Shared cumulative-cost budget over a subset of agents.

    Args:
        amount: Maximum allowed cumulative cost for this budget.
        agents: Subset of agents from ``env.possible_agents`` covered by the budget.
        name: Optional metric prefix. If omitted, a generated name is used.

    Notes:
        Agent memberships are deduplicated while preserving order. Budgets may
        overlap, so a single agent may contribute to more than one budget.
    """

    amount: float
    agents: tuple[str, ...]
    name: str | None = None

    def __post_init__(self):
        agents = tuple(dict.fromkeys(self.agents))
        if not agents:
            raise ValueError("Budget agents must be non-empty.")
        object.__setattr__(self, "amount", float(self.amount))
        object.__setattr__(self, "agents", agents)


class ConstrainedMarkovGame:
    """Cumulative-cost monitor for a labelled parallel PettingZoo environment."""

    def __init__(
        self,
        possible_agents: Sequence[str],
        budgets: Sequence[Budget],
        cost_fn: CostFn = dummy_cost_fn,
    ):
        self.possible_agents = tuple(possible_agents)
        self._possible_agent_set = set(self.possible_agents)
        self.budgets = tuple(budgets)
        self.cost_fn = cost_fn

        if not self.possible_agents:
            raise ValueError("possible_agents must be non-empty.")
        if not self.budgets:
            raise ValueError("budgets must be non-empty.")

        self._budget_keys = []
        seen_budget_keys = set()
        for index, budget in enumerate(self.budgets):
            invalid_agents = set(budget.agents) - self._possible_agent_set
            if invalid_agents:
                raise ValueError(
                    f"Budget agents must exist in possible_agents. Invalid agents: {sorted(invalid_agents)}"
                )
            key = budget.name or f"budget_{index}"
            if key in seen_budget_keys:
                raise ValueError(f"Budget names must be unique. Duplicate name: {key}")
            seen_budget_keys.add(key)
            self._budget_keys.append(key)
        self._budget_keys = tuple(self._budget_keys)

        self.reset()

    def reset(self):
        """Reset per-agent and per-budget cumulative costs for a new episode."""
        self.agent_step_costs = {agent: 0.0 for agent in self.possible_agents}
        self.agent_totals = {agent: 0.0 for agent in self.possible_agents}
        self.budget_step_costs = {key: 0.0 for key in self._budget_keys}
        self.budget_totals = {key: 0.0 for key in self._budget_keys}

    def update(self, labels_by_agent: Mapping[str, Iterable[str]]):
        """Update the monitor from a mapping of agent ids to active labels."""
        unknown_agents = set(labels_by_agent) - self._possible_agent_set
        if unknown_agents:
            raise ValueError(f"Unknown agents in labels_by_agent: {sorted(unknown_agents)}")

        for agent in self.possible_agents:
            labels = labels_by_agent.get(agent, set())
            if not isinstance(labels, (set, frozenset)):
                raise ValueError(
                    f"Expected labels for agent '{agent}' to be a set of atomic propositions, "
                    f"got {type(labels).__name__}"
                )
            step_cost = float(self.cost_fn(labels))
            self.agent_step_costs[agent] = step_cost
            self.agent_totals[agent] += step_cost

        for key, budget in zip(self._budget_keys, self.budgets):
            step_cost = float(sum(self.agent_step_costs[agent] for agent in budget.agents))
            self.budget_step_costs[key] = step_cost
            self.budget_totals[key] += step_cost

    def satisfied(self) -> bool:
        """Return ``True`` when every budget remains within its cap."""
        return all(
            self.budget_totals[key] <= budget.amount
            for key, budget in zip(self._budget_keys, self.budgets)
        )

    def step_metric(self) -> dict[str, float]:
        """Return per-step metrics for agents and budgets."""
        metrics: dict[str, float] = {}
        for agent in self.possible_agents:
            metrics[f"{agent}_cost"] = self.agent_step_costs[agent]
            metrics[f"{agent}_violation"] = float(self.agent_step_costs[agent] >= 0.5)
            metrics[f"{agent}_cum_cost"] = self.agent_totals[agent]
        for key, budget in zip(self._budget_keys, self.budgets):
            metrics[f"{key}_cost"] = self.budget_step_costs[key]
            metrics[f"{key}_cum_cost"] = self.budget_totals[key]
            metrics[f"{key}_satisfied"] = float(self.budget_totals[key] <= budget.amount)
        metrics["satisfied"] = float(self.satisfied())
        return metrics

    def episode_metric(self) -> dict[str, float]:
        """Return end-of-episode cumulative metrics for agents and budgets."""
        metrics: dict[str, float] = {}
        for agent in self.possible_agents:
            metrics[f"{agent}_cum_cost"] = self.agent_totals[agent]
        for key, budget in zip(self._budget_keys, self.budgets):
            metrics[f"{key}_cum_cost"] = self.budget_totals[key]
            metrics[f"{key}_satisfied"] = float(self.budget_totals[key] <= budget.amount)
        metrics["satisfied"] = float(self.satisfied())
        return metrics

    @property
    def constraint_type(self) -> str:
        return "cmg"


class ConstrainedMarkovGameEnv(ParallelEnv):
    """PettingZoo parallel wrapper that updates a :class:`ConstrainedMarkovGame`."""

    def __init__(
        self,
        env: ParallelEnv,
        budgets: Sequence[Budget],
        cost_fn: CostFn = dummy_cost_fn,
        **kw: Any,
    ):
        if not isinstance(env, LabelledParallelEnv):
            raise TypeError(
                f"{self.__class__.__name__} must wrap a LabelledParallelEnv, but got {type(env).__name__}."
            )
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = tuple(env.possible_agents)
        self.agents = list(getattr(env, "agents", self.possible_agents))
        self.label_fn = getattr(env, "label_fn", None)
        self.cost_fn = cost_fn
        self._constraint = ConstrainedMarkovGame(
            possible_agents=self.possible_agents,
            budgets=budgets,
            cost_fn=cost_fn,
        )

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the wrapped env and seed the constraint from initial agent labels."""
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(getattr(self.env, "agents", self.possible_agents))
        self._constraint.reset()
        self._constraint.update(self._labels_by_agent(infos))
        return obs, infos

    def step(self, actions):
        """Step the wrapped env and update the constraint from per-agent labels."""
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = list(getattr(self.env, "agents", self.possible_agents))
        self._constraint.update(self._labels_by_agent(infos))
        return obs, rewards, terminations, truncations, infos

    def state(self):
        return self.env.state()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def constraint_step_metrics(self) -> dict[str, float]:
        return self._constraint.step_metric()

    def constraint_episode_metrics(self) -> dict[str, float]:
        return self._constraint.episode_metric()

    @property
    def constraint_type(self) -> str:
        return self._constraint.constraint_type

    def _labels_by_agent(self, infos: Mapping[str, Mapping[str, Any] | None]) -> dict[str, set[str] | frozenset[str]]:
        """Extract and validate ``infos[agent]['labels']`` for all possible agents."""
        unknown_agents = set(infos) - set(self.possible_agents)
        if unknown_agents:
            raise ValueError(f"Unknown agents in infos: {sorted(unknown_agents)}")

        labels_by_agent: dict[str, set[str] | frozenset[str]] = {}
        for agent in self.possible_agents:
            agent_info = infos.get(agent, {})
            if agent_info is None:
                agent_info = {}
            if not isinstance(agent_info, Mapping):
                raise ValueError(
                    f"Expected info for agent '{agent}' to be a mapping, got {type(agent_info).__name__}"
                )
            labels = agent_info.get("labels", set())
            if not isinstance(labels, (set, frozenset)):
                raise ValueError(
                    f"Expected labels for agent '{agent}' to be a set of atomic propositions, "
                    f"got {type(labels).__name__}"
                )
            labels_by_agent[agent] = labels
        return labels_by_agent
