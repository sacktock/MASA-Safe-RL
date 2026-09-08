"""Support aggregation and decentralised action rectangles for coalitions."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from masa.envs.multiagent.base import TabularParallelEnv


@dataclass(frozen=True, slots=True)
class CoalitionSupport:
    """Qualitative support after universally quantifying complement actions.

    ``legal_actions[state][agent_index]`` snapshots primitive availability at
    synthesis time. ``full_successors[state][joint_action_index]`` stores the
    original support of each legal full joint action. Empty entries denote
    state-illegal joint actions. Both are retained for runtime validation.
    """

    coalition_agents: tuple[str, ...]
    complement_agents: tuple[str, ...]
    coalition_actions: tuple[tuple[int, ...], ...]
    legal_actions: tuple[tuple[tuple[int, ...], ...], ...]
    full_successors: tuple[tuple[tuple[int, ...], ...], ...]
    successors: np.ndarray
    support: np.ndarray


def _legal_actions(
    env: TabularParallelEnv, state: int, agent: str
) -> tuple[int, ...]:
    """Read one validated state-dependent primitive action set."""
    return env.get_legal_actions(state, agent)


def _joint_action(
    all_agents: tuple[str, ...],
    coalition_agents: tuple[str, ...],
    coalition_action: tuple[int, ...],
    complement_agents: tuple[str, ...],
    complement_action: tuple[int, ...],
) -> tuple[int, ...]:
    selected = dict(zip(coalition_agents, coalition_action))
    selected.update(zip(complement_agents, complement_action))
    return tuple(selected[agent] for agent in all_agents)


def build_coalition_support(
    env: TabularParallelEnv,
    coalition_agents: Sequence[str],
) -> CoalitionSupport:
    """Union successor support over every legal complement joint action.

    A coalition action is enabled only when every constituent primitive action is
    legal. Every legal action of each agent outside the coalition contributes to
    its support. Passing the resulting arrays to the ordinary winning-region
    solver therefore implements

    ``exists coalition action, forall complement actions, forall successors``.
    """
    if not isinstance(env, TabularParallelEnv):
        raise TypeError("env must be a TabularParallelEnv.")
    if isinstance(coalition_agents, (str, bytes)):
        raise TypeError("coalition_agents must be a sequence of agent names.")
    requested = tuple(coalition_agents)
    if not requested:
        raise ValueError("coalition_agents must be non-empty.")
    if len(set(requested)) != len(requested):
        raise ValueError("coalition_agents must be unique.")

    all_agents = tuple(env.possible_agents)
    if not all_agents or len(set(all_agents)) != len(all_agents):
        raise ValueError("possible_agents must be non-empty and unique.")
    action_sizes = env.action_sizes
    selected = set(requested)
    unknown = selected - set(all_agents)
    if unknown:
        raise ValueError(f"Unknown coalition agents: {sorted(unknown)}.")

    coalition = tuple(agent for agent in all_agents if agent in selected)
    complement = tuple(agent for agent in all_agents if agent not in selected)
    coalition_actions = tuple(
        product(*(range(action_sizes[agent]) for agent in coalition))
    )

    rows: list[tuple[tuple[int, ...], np.ndarray]] = []
    legal_rows: list[tuple[tuple[int, ...], ...]] = []
    full_rows: list[tuple[tuple[int, ...], ...]] = []
    for state in range(env.n_states):
        legal = {agent: _legal_actions(env, state, agent) for agent in all_agents}
        legal_rows.append(tuple(legal[agent] for agent in all_agents))
        complement_actions = tuple(product(*(legal[a] for a in complement)))
        # product() over no iterables yields one empty tuple for a grand coalition.

        per_coalition_action: list[tuple[int, ...]] = []
        row_successors: set[int] = set()
        full_successors: list[tuple[int, ...]] = [
            () for _ in range(env.n_joint_actions)
        ]

        for coalition_action in coalition_actions:
            if any(
                primitive not in legal[agent]
                for agent, primitive in zip(coalition, coalition_action)
            ):
                per_coalition_action.append(())
                continue

            action_successors: set[int] = set()
            for complement_action in complement_actions:
                full_action = _joint_action(
                    all_agents,
                    coalition,
                    coalition_action,
                    complement,
                    complement_action,
                )
                full_index = env.encode_joint_action(full_action)
                support = full_successors[full_index]
                if not support:
                    support = env.successors(state, full_action)
                    full_successors[full_index] = support
                action_successors.update(support)

            ordered = tuple(sorted(action_successors))
            if not ordered:
                raise ValueError(
                    f"No successor for state {state}, coalition action "
                    f"{coalition_action}."
                )
            per_coalition_action.append(ordered)
            row_successors.update(ordered)

        ids = tuple(sorted(row_successors))
        index = {successor: offset for offset, successor in enumerate(ids)}
        positive = np.zeros((len(coalition_actions), len(ids)), dtype=bool)
        for action_index, action_successors in enumerate(per_coalition_action):
            for successor in action_successors:
                positive[action_index, index[successor]] = True
        rows.append((ids, positive))
        full_rows.append(tuple(full_successors))

    width = max(1, max(len(ids) for ids, _ in rows))
    successors = np.zeros((env.n_states, width), dtype=np.intp)
    support = np.zeros(
        (env.n_states, len(coalition_actions), width), dtype=bool
    )
    for state, (ids, positive) in enumerate(rows):
        successors[state, : len(ids)] = ids
        support[state, :, : len(ids)] = positive

    successors.flags.writeable = False
    support.flags.writeable = False
    return CoalitionSupport(
        coalition_agents=coalition,
        complement_agents=complement,
        coalition_actions=coalition_actions,
        legal_actions=tuple(legal_rows),
        full_successors=tuple(full_rows),
        successors=successors,
        support=support,
    )


def _rectangle_is_safe(
    factors: list[set[int]],
    safe_row: np.ndarray,
    action_index: dict[tuple[int, ...], int],
) -> bool:
    return all(
        safe_row[action_index[tuple(joint_action)]]
        for joint_action in product(*(sorted(factor) for factor in factors))
    )


def rectangular_action_masks(
    safe_joint_actions: np.ndarray,
    coalition_actions: Sequence[tuple[int, ...]],
    action_sizes: Sequence[int],
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Choose sound Cartesian masks for simultaneous independent execution.

    Each returned row is a non-empty Cartesian subset of the corresponding safe
    joint-action relation whenever that relation is non-empty. Coalition members
    may therefore choose independently without observing teammates' current
    actions. Safety relies on teammates respecting their masks; non-coalition
    agents have already been universally quantified.

    The deterministic selector tries each safe tuple as a singleton seed, greedily
    admits primitive actions in agent/action order while preserving Cartesian
    closure, and retains the candidate with the most joint profiles, then the
    largest sum of local mask sizes. It is sound, but not guaranteed to find a
    globally maximum rectangle.
    """
    safe = np.asarray(safe_joint_actions, dtype=bool)
    actions = tuple(tuple(action) for action in coalition_actions)
    sizes = tuple(int(size) for size in action_sizes)
    if safe.ndim != 2 or safe.shape[1] != len(actions):
        raise ValueError("safe_joint_actions has incompatible shape.")
    if not actions or not sizes or len(actions[0]) != len(sizes):
        raise ValueError("coalition actions and action_sizes are incompatible.")
    if any(size <= 0 for size in sizes):
        raise ValueError("action_sizes must be positive.")

    expected = tuple(product(*(range(size) for size in sizes)))
    if actions != expected:
        raise ValueError(
            "coalition_actions must be the canonical Cartesian product order."
        )
    action_index = {action: index for index, action in enumerate(actions)}

    local_masks = tuple(
        np.zeros((safe.shape[0], size), dtype=bool) for size in sizes
    )
    rectangle_joint = np.zeros_like(safe)

    for state in range(safe.shape[0]):
        safe_indices = np.flatnonzero(safe[state])
        if not safe_indices.size:
            continue

        best: list[set[int]] | None = None
        best_score = (-1, -1)
        for seed_index in safe_indices:
            seed = actions[int(seed_index)]
            factors = [{primitive} for primitive in seed]

            for agent_index, size in enumerate(sizes):
                for primitive in range(size):
                    if primitive in factors[agent_index]:
                        continue
                    trial = [set(factor) for factor in factors]
                    trial[agent_index].add(primitive)
                    if _rectangle_is_safe(trial, safe[state], action_index):
                        factors = trial

            score = (
                int(np.prod([len(factor) for factor in factors])),
                sum(len(factor) for factor in factors),
            )
            if score > best_score:
                best = factors
                best_score = score

        assert best is not None
        for agent_index, factor in enumerate(best):
            local_masks[agent_index][state, sorted(factor)] = True
        for joint_action in product(*(sorted(factor) for factor in best)):
            index = action_index[tuple(joint_action)]
            if not safe[state, index]:
                raise AssertionError("Internal error: unsafe rectangle selected.")
            rectangle_joint[state, index] = True

    for mask in local_masks:
        mask.flags.writeable = False
    rectangle_joint.flags.writeable = False
    return local_masks, rectangle_joint
