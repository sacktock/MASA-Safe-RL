"""Independent tabular Q-learning for PettingZoo parallel environments.

``IQL`` is a coordinator around MASA's existing single-agent
:class:`~masa.algorithms.tabular.q_learning.QL` implementation.  It owns one
``QL`` learner per PettingZoo agent, asks every active learner for an action,
steps the shared parallel environment exactly once, and then feeds each learner
its own transition.

The Q-learning target, exploration rule, and epsilon schedule are therefore not
reimplemented here.  Only the multi-agent rollout coordination is new.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from operator import index
from typing import Any, TypeAlias

import gymnasium as gym
import jax.random as jr
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from masa.algorithms.tabular.q_learning import QL
from masa.common.constraints.multi_agent.cmg import Budget

StateEncoder: TypeAlias = Callable[[Any], int]
StateCounts: TypeAlias = int | Mapping[str, int]
StateEncoders: TypeAlias = StateEncoder | Mapping[str, StateEncoder]


class _AgentSpaces(gym.Env):
    """Spaces-only environment view required by ``QL.__init__``.

    A learner must never advance the shared game independently.  ``reset`` and
    ``step`` therefore fail loudly; :class:`IQL` is the only object allowed to
    interact with the PettingZoo environment.
    """

    metadata: dict[str, Any] = {}

    def __init__(self, n_states: int, action_space: spaces.Discrete):
        super().__init__()
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        raise RuntimeError(
            "Use the IQL coordinator, not an individual learner's "
            "rollout/train method."
        )

    def step(self, action: int):
        raise RuntimeError("Only the IQL coordinator may step the shared ParallelEnv.")


class _ExternalQL(QL):
    """Small adapter that lets a coordinator supply transitions to ``QL``."""

    def observe(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminal: bool,
    ) -> None:
        """Apply MASA's existing one-step Q-learning update to one transition."""
        # QL's buffer layout is
        # (state, action, reward, cost, violation, next_state, terminal).
        # IQL variants shape the reward before this point, so the cost fields are
        # deliberately neutral and cannot apply a second penalty.
        self.buffer.append(
            (state, action, float(reward), 0.0, False, next_state, bool(terminal))
        )
        self.optimize(self._step)

        # QL.rollout normally advances these fields.  We bypass rollout because
        # only the coordinator may call ParallelEnv.step().
        self._step += 1
        self._epsilon = self._epsilon_decay_schedule(self._step)


def _as_agent_mapping(
    value: Any,
    agents: Sequence[str],
    *,
    name: str,
) -> dict[str, Any]:
    """Broadcast one value or validate an exact per-agent mapping."""
    if isinstance(value, Mapping):
        missing = set(agents) - set(value)
        unknown = set(value) - set(agents)
        if missing or unknown:
            raise ValueError(
                f"{name} must contain exactly the possible agents; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        return {agent: value[agent] for agent in agents}
    return {agent: value for agent in agents}


def _cmg_budgets(env: ParallelEnv) -> tuple[Budget, ...]:
    """Return CMG budget definitions through the public API, with compatibility fallback."""
    budgets = getattr(env, "budgets", None)
    if budgets is None:
        # Compatibility with MASA versions preceding the public ``budgets``
        # property on ConstrainedMarkovGameEnv.
        constraint = getattr(env, "_constraint", None)
        budgets = getattr(constraint, "budgets", None)
    if budgets is None:
        raise TypeError("The CMG environment must expose its budget definitions.")
    return tuple(budgets)


class IQL:
    """Independent Q-learning using one MASA ``QL`` instance per agent.

    Args:
        env: A MASA constrained Markov game using the PettingZoo parallel API.
            The caller retains ownership of the environment.
        horizon: Maximum number of decisions in one finite episode.  Elapsed
            time is included in each learner's tabular state so that a finite
            horizon game remains Markov.  The environment must terminate or
            truncate every active agent by this limit.
        n_states: Number of encoded observations.  Supply one integer for a
            homogeneous game or a mapping keyed by agent.  When omitted, every
            agent must have a zero-based ``Discrete`` observation space and its
            size is inferred.
        encode: Function mapping one observation to a zero-based integer, or a
            mapping of functions keyed by agent.  Defaults to ``operator.index``.
        seed: Base seed.  Each agent receives an independent learner seed and
            independent action-sampling stream.
        ql_kwargs: Keyword arguments forwarded unchanged to each ``QL`` learner.

    Notes:
        This class is an independent/decentralised learner, not a joint-action
        learner.  Every policy owns a distinct Q-table and uses only the
        observation supplied to that agent by the environment.  If the base
        environment supplies global observations, as ``ChickenMatrix`` does,
        those observations remain global.
    """

    def __init__(
        self,
        env: ParallelEnv,
        *,
        horizon: int,
        n_states: StateCounts | None = None,
        encode: StateEncoders | None = None,
        seed: int = 0,
        ql_kwargs: Mapping[str, Any] | None = None,
    ):
        self.horizon = index(horizon)
        if self.horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if getattr(env, "constraint_type", None) != "CMG":
            raise TypeError("IQL expects a MASA CMG ParallelEnv.")

        self.env = env
        self.seed = index(seed)
        self.agents = tuple(env.possible_agents)
        if not self.agents:
            raise ValueError("The environment must expose at least one possible agent.")

        budget_definitions = _cmg_budgets(env)
        self.budgets = {
            budget.name or f"budget_{position}": budget
            for position, budget in enumerate(budget_definitions)
        }
        if len(self.budgets) != len(budget_definitions):
            raise ValueError("CMG budget names must be unique.")

        self._n_states, self._encoders = self._resolve_state_encoders(
            n_states=n_states,
            encode=encode,
        )

        config = dict(ql_kwargs or {})
        reserved = {"env", "seed"}.intersection(config)
        if reserved:
            raise ValueError(
                f"Pass {sorted(reserved)} to IQL rather than through ql_kwargs."
            )

        self.learners: dict[str, _ExternalQL] = {}
        for position, agent in enumerate(self.agents):
            action_space = env.action_space(agent)
            if not isinstance(action_space, spaces.Discrete) or int(action_space.start) != 0:
                raise TypeError(
                    f"Agent {agent!r} needs a zero-based Discrete action space; "
                    f"got {action_space!r}."
                )

            # Decision times are 0, ..., horizon - 1.  Terminal next states are
            # never indexed, so exactly horizon blocks are needed.
            learner_env = _AgentSpaces(
                self._n_states[agent] * self.horizon,
                action_space,
            )
            self.learners[agent] = _ExternalQL(
                learner_env,
                seed=self.seed + position,
                **config,
            )

        # Kept on the base class so logging and downstream tooling can treat all
        # variants uniformly.  IQL has no active penalties.
        self.lambdas: dict[str, float] = {}
        self.episodes = 0

    def _resolve_state_encoders(
        self,
        *,
        n_states: StateCounts | None,
        encode: StateEncoders | None,
    ) -> tuple[dict[str, int], dict[str, StateEncoder]]:
        if n_states is None:
            inferred: dict[str, int] = {}
            for agent in self.agents:
                observation_space = self.env.observation_space(agent)
                if (
                    not isinstance(observation_space, spaces.Discrete)
                    or int(observation_space.start) != 0
                ):
                    raise TypeError(
                        "n_states is required unless every observation space is "
                        "zero-based Discrete."
                    )
                inferred[agent] = int(observation_space.n)
            state_counts = inferred
        else:
            raw_counts = _as_agent_mapping(n_states, self.agents, name="n_states")
            state_counts = {}
            for agent, raw_count in raw_counts.items():
                count = index(raw_count)
                if count < 1:
                    raise ValueError(f"n_states[{agent!r}] must be positive.")
                state_counts[agent] = count

        if encode is None:
            encoders = {agent: index for agent in self.agents}
        else:
            raw_encoders = _as_agent_mapping(encode, self.agents, name="encode")
            encoders = {}
            for agent, encoder in raw_encoders.items():
                if not callable(encoder):
                    raise TypeError(f"encode[{agent!r}] must be callable.")
                encoders[agent] = encoder

        return state_counts, encoders

    def _state(self, agent: str, observation: Any, time: int) -> int:
        """Encode ``(elapsed time, local observation)`` into one table index."""
        if agent not in self.learners:
            raise KeyError(f"Unknown agent {agent!r}.")
        time = index(time)
        if not 0 <= time < self.horizon:
            raise ValueError(
                f"Decision time {time} is outside the configured horizon "
                f"[0, {self.horizon - 1}]."
            )

        observation_id = index(self._encoders[agent](observation))
        count = self._n_states[agent]
        if not 0 <= observation_id < count:
            raise ValueError(
                f"Encoded observation {observation_id} for {agent!r} is outside "
                f"[0, {count - 1}]."
            )
        return time * count + observation_id

    def _penalty(self, agent: str, metrics: Mapping[str, float]) -> float:
        """Return this variant's immediate CMG penalty for one agent."""
        return 0.0

    def _after_episode(self, metrics: Mapping[str, float]) -> None:
        """Hook for adaptive variants; fixed IQL performs no episode update."""

    def _episode(
        self,
        *,
        training: bool,
        seed: int,
        deterministic: bool,
    ) -> dict[str, float]:
        observations, _ = self.env.reset(seed=seed)
        if not self.env.agents:
            raise ValueError("A training episode must start with a live agent.")

        # Local keys ensure evaluation cannot perturb any learner's future
        # training stream.  All agents still sample independently.
        keys = {
            agent: jr.fold_in(jr.PRNGKey(seed), position)
            for position, agent in enumerate(self.agents)
        }
        returns = {agent: 0.0 for agent in self.agents}
        shaped_returns = {agent: 0.0 for agent in self.agents}
        used_lambdas = self.lambdas.copy()
        time = 0

        while self.env.agents:
            if time >= self.horizon:
                raise ValueError(
                    "The environment remained active past horizon; configure the "
                    "environment to terminate or truncate by the supplied horizon."
                )

            # Snapshot before step(), because ParallelEnv implementations may
            # mutate env.agents when the episode ends.
            active_agents = tuple(self.env.agents)
            states = {
                agent: self._state(agent, observations[agent], time)
                for agent in active_agents
            }

            actions: dict[str, int] = {}
            # Complete the joint action before applying any learner update.
            for agent in active_agents:
                keys[agent], action_key = jr.split(keys[agent])
                actions[agent] = self.learners[agent].act(
                    action_key,
                    states[agent],
                    deterministic=deterministic,
                )

            next_observations, rewards, terminations, truncations, _ = self.env.step(actions)
            step_metrics = self.env.constraint_step_metrics()
            next_time = time + 1

            for agent in active_agents:
                terminal = bool(
                    terminations.get(agent, False) or truncations.get(agent, False)
                )
                if not terminal and agent not in self.env.agents:
                    raise ValueError(f"Agent {agent!r} disappeared without a done flag.")
                if agent not in rewards:
                    raise ValueError(f"ParallelEnv omitted reward for active agent {agent!r}.")

                reward = float(rewards[agent])
                shaped_reward = reward - self._penalty(agent, step_metrics)
                returns[agent] += reward
                shaped_returns[agent] += shaped_reward

                if training:
                    # PettingZoo may omit terminal observations.  QL ignores the
                    # supplied next state whenever terminal=True.
                    next_state = (
                        0
                        if terminal
                        else self._state(agent, next_observations[agent], next_time)
                    )
                    self.learners[agent].observe(
                        states[agent],
                        actions[agent],
                        shaped_reward,
                        next_state,
                        terminal,
                    )

            observations = next_observations
            time = next_time

        episode_metrics = dict(self.env.constraint_episode_metrics())
        if training:
            self._after_episode(episode_metrics)

        row: dict[str, float] = {"steps": float(time), **episode_metrics}
        row.update({f"return/{agent}": value for agent, value in returns.items()})
        row.update(
            {
                f"shaped_return/{agent}": value
                for agent, value in shaped_returns.items()
            }
        )
        # ``lambda/...`` is the value used for this rollout.  ``lambda_next/...``
        # records an episode-end dual update, when present.
        row.update({f"lambda/{name}": value for name, value in used_lambdas.items()})
        row.update({f"lambda_next/{name}": value for name, value in self.lambdas.items()})
        return row

    def train(self, episodes: int) -> list[dict[str, float]]:
        """Train for complete episodes and return one flat metric row per episode."""
        episodes = index(episodes)
        if episodes < 0:
            raise ValueError("episodes must be nonnegative.")

        history: list[dict[str, float]] = []
        for _ in range(episodes):
            row = self._episode(
                training=True,
                seed=self.seed + self.episodes,
                deterministic=False,
            )
            self.episodes += 1
            row["episode"] = float(self.episodes)
            history.append(row)
        return history

    def evaluate(
        self,
        episodes: int = 100,
        *,
        seed: int = 10_000,
        deterministic: bool = True,
    ) -> list[dict[str, float]]:
        """Run evaluation without changing Q-tables, schedules, or multipliers."""
        episodes = index(episodes)
        if episodes < 0:
            raise ValueError("episodes must be nonnegative.")
        seed = index(seed)

        return [
            self._episode(
                training=False,
                seed=seed + episode,
                deterministic=deterministic,
            )
            for episode in range(episodes)
        ]

    @property
    def q_tables(self) -> dict[str, np.ndarray]:
        """Return the live Q-table owned by each independent learner."""
        return {agent: learner.Q for agent, learner in self.learners.items()}

    def close(self) -> None:
        """Close the coordinated environment."""
        self.env.close()


__all__ = ["IQL", "StateEncoder", "StateCounts", "StateEncoders"]
