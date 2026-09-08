"""Base class for finite PettingZoo parallel environments."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

JointAction = tuple[int, ...]


class TabularParallelEnv(ParallelEnv):
    """Parallel environment with an enumerable finite-state transition model.

    Subclasses set ``_n_states`` and expose either:

    * ``_transition_matrix[next_state, state, joint_action_index]``; or
    * ``_successor_states[state]`` together with
      ``_transition_probs[state, joint_action_index]``.

    Joint-action indices use the Cartesian-product order induced by
    ``possible_agents`` and their zero-based discrete action spaces. For example,
    two binary agents use ``(0, 0), (0, 1), (1, 0), (1, 1)``.

    ``get_state_id()`` returns the current tabular state. By default this reads
    ``self._state``. ``observations_from_state()`` reconstructs the per-agent
    observations used by :class:`~masa.common.multi_agent.LabelledParallelEnv`
    when synthesising an LTL product. The default implementation supports the
    common case where every agent directly observes the same discrete state ID.

    The class deliberately does not define rewards, labels, reset, or step. It
    only standardises the finite game model needed by planning and shielding.
    """

    def __init__(self) -> None:
        super().__init__()
        self._n_states: int | None = None
        self._transition_matrix: np.ndarray | None = None
        self._successor_states: Mapping[int, Sequence[int]] | None = None
        self._transition_probs: Mapping[tuple[int, int], Sequence[float]] | None = None
        self._state: int | None = None

    @property
    def n_states(self) -> int:
        """Number of finite model states."""
        value = self._n_states
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError("_n_states must be set to a positive integer.")
        value = int(value)
        if value <= 0:
            raise ValueError("_n_states must be positive.")
        return value

    @property
    def action_sizes(self) -> dict[str, int]:
        """Primitive action count for each agent in ``possible_agents`` order."""
        agents = tuple(self.possible_agents)
        if not agents or len(set(agents)) != len(agents):
            raise ValueError("possible_agents must be non-empty and unique.")

        sizes: dict[str, int] = {}
        for agent in agents:
            space = self.action_space(agent)
            if not isinstance(space, spaces.Discrete) or int(space.start) != 0:
                raise TypeError(
                    "TabularParallelEnv requires zero-based Discrete action spaces."
                )
            sizes[agent] = int(space.n)
        return sizes

    @property
    def n_joint_actions(self) -> int:
        """Size of the full Cartesian joint-action space."""
        return int(np.prod(tuple(self.action_sizes.values()), dtype=np.int64))

    @property
    def has_transition_matrix(self) -> bool:
        return self._transition_matrix is not None

    @property
    def has_successor_states_dict(self) -> bool:
        return (
            self._successor_states is not None
            and self._transition_probs is not None
        )

    def get_transition_matrix(self) -> np.ndarray | None:
        """Return ``P[next_state, state, joint_action_index]``, when available."""
        return self._transition_matrix

    def get_successor_states_dict(
        self,
    ) -> tuple[
        Mapping[int, Sequence[int]],
        Mapping[tuple[int, int], Sequence[float]],
    ] | None:
        """Return the sparse state-successor/probability representation.

        Probability vectors are aligned with ``successor_states[state]`` and
        keyed by ``(state, joint_action_index)``.
        """
        if not self.has_successor_states_dict:
            return None
        assert self._successor_states is not None
        assert self._transition_probs is not None
        return self._successor_states, self._transition_probs

    def _check_state(self, state: int) -> int:
        if not isinstance(state, Integral) or isinstance(state, bool):
            raise TypeError(f"State ID must be an integer, got {state!r}.")
        state = int(state)
        if not 0 <= state < self.n_states:
            raise ValueError(
                f"State ID {state} is outside [0, {self.n_states})."
            )
        return state

    def _joint_action_tuple(
        self, actions: Mapping[str, int] | Sequence[int]
    ) -> JointAction:
        agents = tuple(self.possible_agents)
        if isinstance(actions, Mapping):
            supplied = set(actions)
            expected = set(agents)
            if supplied != expected:
                raise ValueError(
                    "Joint action must contain exactly possible_agents; "
                    f"missing={sorted(expected - supplied)}, "
                    f"extra={sorted(supplied - expected)}."
                )
            raw = tuple(actions[agent] for agent in agents)
        else:
            if isinstance(actions, (str, bytes)):
                raise TypeError("Joint action must be a mapping or action sequence.")
            raw = tuple(actions)
            if len(raw) != len(agents):
                raise ValueError(
                    f"Expected {len(agents)} primitive actions, got {len(raw)}."
                )

        sizes = self.action_sizes
        result: list[int] = []
        for agent, primitive in zip(agents, raw):
            if not isinstance(primitive, Integral) or isinstance(primitive, bool):
                raise TypeError(
                    f"Action for {agent!r} must be an integer, got {primitive!r}."
                )
            primitive = int(primitive)
            if not 0 <= primitive < sizes[agent]:
                raise ValueError(
                    f"Action {primitive} is outside the action space of {agent!r}."
                )
            result.append(primitive)
        return tuple(result)

    def encode_joint_action(
        self, actions: Mapping[str, int] | Sequence[int]
    ) -> int:
        """Encode a full joint action in canonical Cartesian-product order."""
        action = self._joint_action_tuple(actions)
        index = 0
        for primitive, size in zip(action, self.action_sizes.values()):
            index = index * size + primitive
        return int(index)

    def decode_joint_action(self, index: int) -> dict[str, int]:
        """Decode a canonical joint-action index to an agent-action mapping."""
        if not isinstance(index, Integral) or isinstance(index, bool):
            raise TypeError("Joint-action index must be an integer.")
        index = int(index)
        if not 0 <= index < self.n_joint_actions:
            raise ValueError(
                f"Joint-action index {index} is outside [0, {self.n_joint_actions})."
            )

        sizes = tuple(self.action_sizes.values())
        primitives = [0] * len(sizes)
        remainder = index
        for position in range(len(sizes) - 1, -1, -1):
            remainder, primitives[position] = divmod(remainder, sizes[position])
        return dict(zip(self.possible_agents, primitives))

    def legal_actions(self, state: int, agent: str) -> tuple[int, ...]:
        """Legal primitive actions for an agent in a model state.

        Override for state-dependent action availability. The default permits the
        entire action space.
        """
        self._check_state(state)
        try:
            size = self.action_sizes[agent]
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {agent!r}.") from exc
        return tuple(range(size))

    def get_legal_actions(self, state: int, agent: str) -> tuple[int, ...]:
        """Return a validated, sorted copy of :meth:`legal_actions`."""
        state = self._check_state(state)
        try:
            size = self.action_sizes[agent]
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {agent!r}.") from exc
        raw = tuple(self.legal_actions(state, agent))
        if not raw:
            raise ValueError(
                f"Agent {agent!r} has no legal action in model state {state}."
            )
        if any(
            not isinstance(action, Integral)
            or isinstance(action, bool)
            or not 0 <= int(action) < size
            for action in raw
        ):
            raise ValueError(
                f"Invalid legal actions for {agent!r} in state {state}: {raw!r}."
            )
        return tuple(sorted(set(map(int, raw))))

    def successors(
        self,
        state: int,
        actions: Mapping[str, int] | Sequence[int],
    ) -> tuple[int, ...]:
        """Return every non-zero-probability successor of a legal joint action."""
        state = self._check_state(state)
        action = self._joint_action_tuple(actions)
        for agent, primitive in zip(self.possible_agents, action):
            if primitive not in self.get_legal_actions(state, agent):
                raise ValueError(
                    f"Action {primitive} is not legal for {agent!r} in state {state}."
                )
        action_index = self.encode_joint_action(action)

        if self.has_successor_states_dict:
            sparse = self.get_successor_states_dict()
            assert sparse is not None
            successor_states, transition_probs = sparse
            ids = np.asarray(successor_states.get(state, ()))
            probs = np.asarray(
                transition_probs.get((state, action_index), ()),
                dtype=np.float64,
            )
        elif self.has_transition_matrix:
            matrix = np.asarray(self.get_transition_matrix())
            expected = (self.n_states, self.n_states, self.n_joint_actions)
            if matrix.shape != expected:
                raise ValueError(
                    "Expected transition shape "
                    f"(next_state, state, joint_action)={expected}, got {matrix.shape}."
                )
            ids = np.arange(self.n_states, dtype=np.intp)
            probs = np.asarray(matrix[:, state, action_index], dtype=np.float64)
        else:
            raise ValueError(
                "TabularParallelEnv must expose a transition matrix or sparse "
                "successor-state dictionaries."
            )

        if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
            raise ValueError(f"Successors of state {state} must be integer IDs.")
        ids = ids.astype(np.intp, copy=False)
        if np.any((ids < 0) | (ids >= self.n_states)):
            raise ValueError(f"Successor of state {state} is out of range.")
        if probs.shape != (ids.size,):
            raise ValueError(
                f"Probability vector for state {state}, joint action {action} "
                "has the wrong shape."
            )
        if not np.all(np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError(
                f"Invalid probabilities at state {state}, joint action {action}."
            )
        if not np.isclose(probs.sum(), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Transition probabilities for state {state}, joint action "
                f"{action} must sum to 1."
            )

        # No probability threshold: every positive-probability outcome matters.
        result = tuple(sorted(set(map(int, ids[probs > 0]))))
        if not result:
            raise ValueError(
                f"Missing transition support for state {state}, joint action {action}."
            )
        return result

    def get_state_id(self) -> int:
        """Return the current finite model-state ID.

        Subclasses with a non-integer runtime representation may override this,
        but should still return the exact state indexing the transition model.
        """
        if self._state is None:
            raise RuntimeError("The environment has no active tabular state.")
        return self._check_state(self._state)

    def observations_from_state(self, state: int) -> dict[str, Any]:
        """Return each possible agent's observation for a finite model state.

        The default supports fully observed environments where every agent's
        observation is the same zero-based ``Discrete(n_states)`` state ID.
        Environments with structured or local observations should override it.
        """
        state = self._check_state(state)
        observations: dict[str, Any] = {}
        for agent in self.possible_agents:
            space = self.observation_space(agent)
            if (
                not isinstance(space, spaces.Discrete)
                or int(space.start) != 0
                or int(space.n) != self.n_states
            ):
                raise NotImplementedError(
                    f"{type(self).__name__}.observations_from_state() must be "
                    "implemented for structured or non-state observations."
                )
            observations[agent] = state
        return observations


__all__ = ["JointAction", "TabularParallelEnv"]
