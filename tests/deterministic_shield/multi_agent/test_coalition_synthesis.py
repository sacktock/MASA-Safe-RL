from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from masa.common.ltl import Atom, DFA
from masa.deterministic_shield.multi_agent.coalition_support import (
    build_coalition_support,
    rectangular_action_masks,
)
from masa.deterministic_shield.winning_region import winning_region

from conftest import TinyParallelEnv


def solve(env, coalition, dfa):
    game = build_coalition_support(env, coalition)
    q_states = tuple(dfa.states)
    q_index = {q: i for i, q in enumerate(q_states)}
    next_q = np.empty((len(q_states), env.n_states), dtype=np.intp)
    for qi, q in enumerate(q_states):
        for state in range(env.n_states):
            next_q[qi, state] = q_index[dfa.transition(q, env.labels(state))]
    targets = next_q[:, game.successors] * env.n_states + game.successors
    rejecting = np.array([q in dfa.accepting for q in q_states], dtype=bool)
    winning, safe = winning_region(targets, game.support, rejecting)
    return game, winning, safe


def never_bad():
    dfa = DFA([0, 1], initial=0, accepting=[1])
    dfa.add_edge(0, 1, Atom("bad"))
    return dfa


def env_from_support(agents, n_states, action_sizes, support, labels, **kwargs):
    return TinyParallelEnv(
        agents=agents,
        n_states=n_states,
        action_sizes=action_sizes,
        transitions=support,
        labels=labels,
        **kwargs,
    )


def test_quantifier_order_is_exists_coalition_then_forall_outsider():
    # For each outsider action a matching response exists, but no single
    # coalition action works against both outsider actions.
    support = {}
    for state in range(2):
        for controlled, outsider in product(range(2), repeat=2):
            successor = 1 if state == 1 or controlled != outsider else 0
            support[(state, (controlled, outsider))] = (successor,)
    env = env_from_support(
        ("controlled", "outsider"),
        2,
        {"controlled": 2, "outsider": 2},
        support,
        lambda state: {"bad"} if state == 1 else set(),
    )

    game, winning, safe = solve(env, ("controlled",), never_bad())
    assert game.complement_agents == ("outsider",)
    np.testing.assert_array_equal(game.support[0].any(axis=1), [True, True])
    assert not winning[0]
    assert not safe[0].any()

    _, winning_all, safe_all = solve(
        env, ("controlled", "outsider"), never_bad()
    )
    assert winning_all[0]
    safe_tuples = {
        action
        for action, allowed in zip(product(range(2), repeat=2), safe_all[0])
        if allowed
    }
    assert safe_tuples == {(0, 0), (1, 1)}


def test_rare_unsafe_successor_disqualifies_action():
    support = {
        (0, (0,)): (0,),
        (0, (1,)): (0, 1),
        (1, (0,)): (1,),
        (1, (1,)): (1,),
    }
    env = env_from_support(
        ("agent",),
        2,
        {"agent": 2},
        support,
        lambda state: {"bad"} if state == 1 else set(),
    )
    _, winning, safe = solve(env, ("agent",), never_bad())
    assert winning[0]
    np.testing.assert_array_equal(safe[0], [True, False])


def test_state_dependent_illegal_coalition_action_is_disabled():
    support = {
        (0, (0, 0)): (0,),
        (0, (0, 1)): (0,),
        (1, (0, 0)): (1,),
        (1, (0, 1)): (1,),
        (1, (1, 0)): (1,),
        (1, (1, 1)): (1,),
    }

    def legal(state, agent):
        if state == 0 and agent == "a":
            return (0,)
        return (0, 1)

    env = env_from_support(
        ("a", "b"),
        2,
        {"a": 2, "b": 2},
        support,
        lambda state: set(),
        legal_actions_fn=legal,
    )
    game = build_coalition_support(env, ("a",))
    np.testing.assert_array_equal(game.support[0].any(axis=1), [True, False])


def test_full_joint_support_is_retained_for_runtime_checks():
    support = {
        (0, (0, 0)): (0,),
        (0, (0, 1)): (0, 1),
        (0, (1, 0)): (1,),
        (0, (1, 1)): (0,),
        (1, (0, 0)): (1,),
        (1, (0, 1)): (1,),
        (1, (1, 0)): (1,),
        (1, (1, 1)): (1,),
    }
    env = env_from_support(
        ("a", "b"),
        2,
        {"a": 2, "b": 2},
        support,
        lambda state: set(),
    )
    game = build_coalition_support(env, ("a",))
    assert game.full_successors[0][env.encode_joint_action((0, 1))] == (0, 1)
    assert game.full_successors[0][env.encode_joint_action((1, 0))] == (1,)


def test_decentralised_masks_are_a_safe_cartesian_rectangle():
    actions = tuple(product(range(2), repeat=2))
    safe = np.array([[True, True, True, False]], dtype=bool)
    masks, rectangle = rectangular_action_masks(safe, actions, (2, 2))

    np.testing.assert_array_equal(masks[0][0], [True, True])
    np.testing.assert_array_equal(masks[1][0], [True, False])
    np.testing.assert_array_equal(rectangle[0], [True, False, True, False])
    for a, b in product(np.flatnonzero(masks[0][0]), np.flatnonzero(masks[1][0])):
        assert safe[0, actions.index((int(a), int(b)))]


def test_coordination_only_actions_are_not_independently_exposed():
    actions = tuple(product(range(2), repeat=2))
    safe = np.array([[True, False, False, True]], dtype=bool)
    masks, rectangle = rectangular_action_masks(safe, actions, (2, 2))

    assert int(rectangle[0].sum()) == 1
    assert int(masks[0][0].sum()) == 1
    assert int(masks[1][0].sum()) == 1
    joint = (
        int(np.flatnonzero(masks[0][0])[0]),
        int(np.flatnonzero(masks[1][0])[0]),
    )
    assert joint in {(0, 0), (1, 1)}
    assert rectangle[0, actions.index(joint)]


def test_build_support_rejects_duplicate_coalition_members():
    env = env_from_support(
        ("a", "b"),
        1,
        {"a": 1, "b": 1},
        {(0, (0, 0)): (0,)},
        lambda state: set(),
    )
    with pytest.raises(ValueError, match="unique"):
        build_coalition_support(env, ("a", "a"))


def test_rectangle_selection_is_sound_for_random_relations():
    rng = np.random.default_rng(9321)
    actions = tuple(product(range(3), range(2), range(2)))
    for _ in range(200):
        safe = rng.random((8, len(actions))) < 0.55
        masks, rectangle = rectangular_action_masks(safe, actions, (3, 2, 2))
        for state in range(safe.shape[0]):
            if not safe[state].any():
                assert not rectangle[state].any()
                assert all(not mask[state].any() for mask in masks)
                continue
            assert all(mask[state].any() for mask in masks)
            expected = np.zeros(len(actions), dtype=bool)
            for joint in product(*(np.flatnonzero(mask[state]) for mask in masks)):
                index = actions.index(tuple(map(int, joint)))
                expected[index] = True
                assert safe[state, index]
            np.testing.assert_array_equal(rectangle[state], expected)


def test_solver_matches_set_reference_on_random_games():
    rng = np.random.default_rng(771)
    agents = ("a", "b", "outsider")
    action_sizes = {agent: 2 for agent in agents}
    coalition = ("a", "b")
    coalition_actions = tuple(product(range(2), repeat=2))

    for _ in range(100):
        n_states = 3
        support = {}
        for state in range(n_states):
            for joint in product(range(2), repeat=3):
                count = int(rng.integers(1, 3))
                support[(state, joint)] = tuple(
                    sorted(set(map(int, rng.integers(n_states, size=count))))
                )
        bad = rng.random(n_states) < 0.25
        env = env_from_support(
            agents,
            n_states,
            action_sizes,
            support,
            lambda state, bad=bad: {"bad"} if bad[state] else set(),
        )
        _, got_winning, got_safe = solve(env, coalition, never_bad())

        reference = {state for state in range(n_states) if not bad[state]}
        while True:
            allowed = np.zeros((n_states, len(coalition_actions)), dtype=bool)
            for state in reference:
                for index, coalition_action in enumerate(coalition_actions):
                    allowed[state, index] = all(
                        all(
                            successor in reference
                            for successor in support[(
                                state,
                                (
                                    coalition_action[0],
                                    coalition_action[1],
                                    outsider,
                                ),
                            )]
                        )
                        for outsider in range(2)
                    )
            updated = set(np.flatnonzero(allowed.any(axis=1)))
            if updated == reference:
                break
            reference = updated

        for state in range(n_states):
            q = 1 if bad[state] else 0
            product_state = q * n_states + state
            assert got_winning[product_state] == (state in reference)
            if q == 0:
                np.testing.assert_array_equal(got_safe[product_state], allowed[state])
            else:
                assert not got_safe[product_state].any()
        assert not got_winning[n_states:].any()
