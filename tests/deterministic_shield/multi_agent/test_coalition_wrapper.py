from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from masa.common.ltl import And, Atom, DFA, Neg
from masa.common.multi_agent import Coalition
from masa.deterministic_shield import CoalitionLTLShield, random_safe

from conftest import TinyParallelEnv, bad_dfa, labelled


def make_all_except_11_three_agents():
    agents = ("a", "b", "outsider")
    transitions = {}
    for state in range(2):
        for joint in product(range(2), repeat=3):
            bad = state == 1 or (joint[0] == 1 and joint[1] == 1)
            transitions[(state, joint)] = (int(bad),)
    return TinyParallelEnv(agents=agents, transitions=transitions)


def test_centralised_preemptive_rejects_before_step(all_except_11_env):
    env = all_except_11_env
    shield = CoalitionLTLShield(
        labelled(env),
        coalition=Coalition(("a", "b")),
        dfa=bad_dfa(),
        mode="preemptive",
        execution="centralised",
    )
    _, infos = shield.reset()
    np.testing.assert_array_equal(
        infos["a"]["shield_joint_action_mask"],
        [True, True, True, False],
    )
    assert shield.safe_coalition_actions() == (
        {"a": 0, "b": 0},
        {"a": 0, "b": 1},
        {"a": 1, "b": 0},
    )

    with pytest.raises(ValueError, match="unsafe"):
        shield.step({"a": 1, "b": 1})
    assert env.step_count == 0
    assert shield.product_state == 0

    _, _, _, _, infos = shield.step({"a": 1, "b": 0})
    assert env.last_actions == {"a": 1, "b": 0}
    assert infos["a"]["shield_intervened"] is False
    assert infos["b"]["shield_intervened"] is False


def test_centralised_postposed_changes_only_coalition_actions():
    env = make_all_except_11_three_agents()
    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        mode="postposed",
        execution="centralized",
    )
    shield.reset()
    _, _, _, _, infos = shield.step({"a": 1, "b": 1, "outsider": 1})
    assert env.last_actions == {"a": 0, "b": 0, "outsider": 1}
    assert "shield_intervened" not in infos["outsider"]
    assert infos["a"]["shield_joint_intervened"] is True
    assert infos["a"]["shield_proposed_coalition_action"] == (1, 1)
    assert infos["a"]["shield_executed_coalition_action"] == (0, 0)


def test_centralised_custom_and_random_replacement_are_checked(all_except_11_env):
    env = all_except_11_env
    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        mode="postposed",
        execution="centralised",
        replacement=lambda state, proposed, mask: 2,
    )
    shield.reset()
    shield.step({"a": 1, "b": 1})
    assert env.last_actions == {"a": 1, "b": 0}

    invalid_env = TinyParallelEnv(transitions=dict(env.transitions))
    invalid = CoalitionLTLShield(
        labelled(invalid_env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        mode="postposed",
        execution="centralised",
        replacement=lambda state, proposed, mask: 3,
    )
    invalid.reset()
    with pytest.raises(ValueError, match="not safe"):
        invalid.step({"a": 1, "b": 1})
    assert invalid_env.step_count == 0

    random_env = TinyParallelEnv(transitions=dict(env.transitions))
    random_shield = CoalitionLTLShield(
        labelled(random_env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        mode="postposed",
        execution="centralised",
        replacement=random_safe(seed=7),
    )
    random_shield.reset()
    random_shield.step({"a": 1, "b": 1})
    assert tuple(random_env.last_actions.values()) in {(0, 0), (0, 1), (1, 0)}


def test_decentralised_masks_form_independent_promises(all_except_11_env):
    env = all_except_11_env
    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        mode="preemptive",
        execution="decentralised",
    )
    _, infos = shield.reset()
    np.testing.assert_array_equal(shield.local_action_mask("a"), [True, True])
    np.testing.assert_array_equal(shield.local_action_mask("b"), [True, False])
    np.testing.assert_array_equal(infos["a"]["shield_action_mask"], [True, True])
    np.testing.assert_array_equal(infos["b"]["shield_action_mask"], [True, False])
    assert shield.safe_coalition_actions() == (
        {"a": 0, "b": 0},
        {"a": 1, "b": 0},
    )

    with pytest.raises(ValueError, match="b=1"):
        shield.step({"a": 1, "b": 1})
    assert env.step_count == 0
    shield.step({"a": 1, "b": 0})
    assert env.last_actions == {"a": 1, "b": 0}


def test_decentralised_postposed_does_not_use_teammate_proposals(all_except_11_env):
    env = all_except_11_env
    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        mode="postposed",
        execution="decentralized",
    )
    shield.reset()
    _, _, _, _, infos = shield.step({"a": 1, "b": 1})
    assert env.last_actions == {"a": 1, "b": 0}
    assert infos["a"]["shield_intervened"] is False
    assert infos["b"]["shield_intervened"] is True

    shield.reset()
    shield.step({"a": 0, "b": 1})
    assert env.last_actions == {"a": 0, "b": 0}


def test_decentralised_replacement_mapping_and_outsider_preservation():
    env = make_all_except_11_three_agents()
    calls = {"b": 0}

    def replace_b(state, proposed, mask):
        calls["b"] += 1
        return 0

    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        mode="postposed",
        execution="decentralised",
        replacement={"b": replace_b},
    )
    shield.reset()
    shield.step({"a": 1, "b": 1, "outsider": 1})
    assert env.last_actions == {"a": 1, "b": 0, "outsider": 1}
    assert calls["b"] == 1


def test_temporal_monitor_consumes_initial_then_successor_labels():
    agents = ("a", "outsider")
    transitions = {}
    for state in range(3):
        for action_a, action_out in product(range(2), repeat=2):
            transitions[(state, (action_a, action_out))] = (
                1 if action_a == 0 else 2,
            )
    labels = lambda state: (
        {"request"} if state == 0 else {"grant"} if state == 2 else set()
    )
    env = TinyParallelEnv(
        agents=agents,
        n_states=3,
        transitions=transitions,
        labels=labels,
    )
    dfa = DFA([10, 20, 99], initial=10, accepting=[99])
    dfa.add_edge(10, 20, Atom("request"))
    dfa.add_edge(20, 99, Neg(Atom("grant")))
    dfa.add_edge(20, 10, And(Atom("grant"), Neg(Atom("request"))))

    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a",),
        dfa=dfa,
        mode="preemptive",
        execution="centralised",
    )
    shield.reset()
    assert shield.automaton_state == 20
    np.testing.assert_array_equal(shield.coalition_action_mask(), [False, True])
    shield.step({"a": 1, "outsider": 1})
    assert shield.automaton_state == 10


def test_transition_model_disagreement_fails_closed():
    modeled = {
        (0, (0,)): (0,),
        (1, (0,)): (1,),
    }
    runtime = {
        (0, (0,)): (1,),
        (1, (0,)): (1,),
    }
    env = TinyParallelEnv(
        agents=("a",),
        action_sizes={"a": 1},
        transitions=modeled,
        runtime_transitions=runtime,
        labels=lambda state: set(),
    )
    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a",),
        dfa=bad_dfa(),
    )
    shield.reset()
    with pytest.raises(RuntimeError, match="absent"):
        shield.step({"a": 0})
    with pytest.raises(RuntimeError, match="reset"):
        _ = shield.product_state


def test_runtime_observation_and_tabular_state_labels_must_agree():
    transitions = {(state, (0,)): (state,) for state in range(2)}
    env = TinyParallelEnv(
        agents=("a",),
        action_sizes={"a": 1},
        transitions=transitions,
        labels=lambda state: {"bad"} if state == 1 else set(),
    )
    shield = CoalitionLTLShield(labelled(env), coalition=("a",), dfa=bad_dfa())

    original = env.observations_from_state
    env.observations_from_state = lambda state: {"a": 0}
    # Synthesis happened before the mutation; reset state 1 now emits state-0 labels.
    with pytest.raises(RuntimeError, match="Runtime labels disagree"):
        shield.reset(options={"state": 1})
    env.observations_from_state = original


def test_custom_label_combiner_can_namespace_agent_labels():
    transitions = {
        (0, (0, 0)): (1,),
        (0, (0, 1)): (1,),
        (1, (0, 0)): (1,),
        (1, (0, 1)): (1,),
    }
    env = TinyParallelEnv(
        agents=("a", "b"),
        n_states=2,
        action_sizes={"a": 1, "b": 2},
        transitions=transitions,
        labels=lambda state: {"unsafe"} if state == 1 else set(),
    )

    def namespace(labels_by_agent):
        return {
            f"{agent}:{label}"
            for agent, labels in labels_by_agent.items()
            for label in labels
        }

    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a",),
        dfa=bad_dfa("a:unsafe"),
        label_combiner=namespace,
    )
    with pytest.raises(RuntimeError, match="outside"):
        shield.reset()


def test_losing_reset_and_episode_boundaries(all_except_11_env):
    env = all_except_11_env
    shield = CoalitionLTLShield(labelled(env), coalition=("a", "b"), dfa=bad_dfa())
    with pytest.raises(RuntimeError, match="outside"):
        shield.reset(options={"state": 1})

    truncated_env = TinyParallelEnv(
        transitions=dict(env.transitions), finish="truncated"
    )
    truncated = CoalitionLTLShield(
        labelled(truncated_env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        execution="decentralised",
    )
    truncated.reset()
    _, _, _, truncations, infos = truncated.step({"a": 0, "b": 0})
    assert all(truncations.values())
    assert infos["a"]["shield_action_mask"].any()
    with pytest.raises(RuntimeError, match="reset"):
        truncated.step({"a": 0, "b": 0})

    terminated_env = TinyParallelEnv(
        transitions=dict(env.transitions), finish="terminated"
    )
    terminated = CoalitionLTLShield(
        labelled(terminated_env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        execution="centralised",
    )
    terminated.reset()
    _, _, terminations, _, infos = terminated.step({"a": 0, "b": 0})
    assert all(terminations.values())
    assert not infos["a"]["shield_joint_action_mask"].any()


def test_terminal_info_retains_final_product_state(all_except_11_env):
    env = TinyParallelEnv(
        transitions=dict(all_except_11_env.transitions), finish="terminated"
    )
    shield = CoalitionLTLShield(
        labelled(env),
        coalition=("a", "b"),
        dfa=bad_dfa(),
        execution="centralised",
    )
    shield.reset()
    _, _, terminations, _, infos = shield.step({"a": 0, "b": 0})
    assert all(terminations.values())
    assert infos["a"]["shield_product_state"] == 0
    assert infos["a"]["shield_automaton_state"] == 0
    assert not infos["a"]["shield_joint_action_mask"].any()


def test_decentralised_multiagent_replacement_requires_separate_callbacks(
    all_except_11_env,
):
    env = all_except_11_env
    callback = random_safe(seed=3)
    with pytest.raises(TypeError, match="one replacement callback per agent"):
        CoalitionLTLShield(
            labelled(env),
            coalition=("a", "b"),
            dfa=bad_dfa(),
            mode="postposed",
            execution="decentralised",
            replacement=callback,
        )

    other = TinyParallelEnv(transitions=dict(env.transitions))
    with pytest.raises(ValueError, match="separate replacement callback"):
        CoalitionLTLShield(
            labelled(other),
            coalition=("a", "b"),
            dfa=bad_dfa(),
            mode="postposed",
            execution="decentralised",
            replacement={"a": callback, "b": callback},
        )
