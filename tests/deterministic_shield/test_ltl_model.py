import numpy as np
import pytest

from masa.common.ltl import And, Atom, DFA, Neg
from masa.deterministic_shield import PreemptiveLTLShield, PostposedLTLShield

pytestmark = pytest.mark.parametrize(
    "shield_class", [PreemptiveLTLShield, PostposedLTLShield], indirect=True
)


@pytest.mark.parametrize("sparse", [False, True])
def test_temporal_memory_and_initial_label(shield_class, make_shield, sparse):
    # G(request -> X grant). State names are not consecutive array indices.
    dfa = DFA([99, 10, 20], 10, [99])
    dfa.add_edge(10, 20, Atom("request"))
    dfa.add_edge(20, 99, Neg(Atom("grant")))
    dfa.add_edge(20, 10, And(Atom("grant"), Neg(Atom("request"))))
    model = np.zeros((3, 3, 2))
    model[1, :, 0] = 1  # Neither request nor grant.
    model[2, :, 1] = 1  # Grant.
    env = make_shield(
        shield_class, model=model, dfa=dfa, sparse=sparse,
        labels=[{"request"}, set(), {"grant"}],
    )
    obs, info = env.reset(seed=0)
    q_index = env.env._automaton_states_idx
    assert obs == q_index[20] * 3  # Initial request already consumed.
    np.testing.assert_array_equal(info["action_mask"], [False, True])
    # Same base state, but no pending obligation: both actions are safe.
    np.testing.assert_array_equal(env.safe_actions[q_index[10] * 3], [True, True])
    obs, _, _, _, info = env.step(1)
    assert obs == q_index[10] * 3 + 2
    np.testing.assert_array_equal(info["action_mask"], [True, True])


@pytest.mark.parametrize("sparse", [False, True])
def test_tiny_unsafe_probability_is_not_ignored(shield_class, make_shield, sparse):
    model = np.zeros((2, 2, 2))
    model[0, 0, 0] = 1
    model[0, 0, 1], model[1, 0, 1] = 1 - 1e-14, 1e-14
    model[1, 1, :] = 1
    env = make_shield(
        shield_class, model=model, labels=[set(), {"bad"}], sparse=sparse
    )
    env.reset()
    np.testing.assert_array_equal(env.action_masks(), [True, False])


@pytest.mark.parametrize("sparse", [False, True])
def test_zero_mass_actions_are_disabled(shield_class, make_shield, sparse):
    model = np.zeros((1, 1, 2))
    model[0, 0, 1] = 1
    env = make_shield(shield_class, model=model, sparse=sparse)
    env.reset()
    np.testing.assert_array_equal(env.action_masks(), [False, True])


@pytest.mark.parametrize("state", [1, 2])
def test_losing_reset_fails_closed(shield_class, make_shield, state):
    env = make_shield(shield_class)
    env.reset()
    with pytest.raises(RuntimeError, match="outside the winning region"):
        env.reset(options={"state": state})
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_deadlock_is_losing(shield_class, make_shield):
    env = make_shield(shield_class, model=np.zeros((1, 1, 2)))
    with pytest.raises(RuntimeError, match="outside the winning region"):
        env.reset()


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_episode_lifecycle(shield_class, make_shield, ending):
    env = make_shield(shield_class, model=np.ones((1, 1, 2)), ending=ending)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.action_masks()
    env.reset()
    _, _, term, trunc, info = env.step(0)
    assert (term, trunc) == (ending == "terminated", ending == "truncated")
    if term:
        assert not info["action_mask"].any()
    else:
        # Truncation ends execution but not the modeled MDP: retain the mask
        # for a masked value target at the returned final observation.
        np.testing.assert_array_equal(info["action_mask"], [True, True])
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.action_masks()
    env.reset()
    env.close()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_unmodeled_transition_invalidates_episode(shield_class, make_shield):
    model = np.zeros((2, 2, 2))
    model[0, 0, :] = 1
    model[1, 1, :] = 1
    env = make_shield(shield_class, model=model)
    env.reset()
    env.unwrapped.actual_model[:, 0, 0] = [0, 1]
    with pytest.raises(RuntimeError, match="absent from the shield model"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_environment_exception_invalidates_episode(shield_class, make_shield):
    env = make_shield(shield_class)
    env.reset()
    env.unwrapped.fail_step = True
    with pytest.raises(RuntimeError, match="simulated environment failure"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.action_masks()


def test_invalid_proposal_does_not_step(shield_class, make_shield):
    env = make_shield(shield_class)
    env.reset()
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)
    assert not env.unwrapped.executed
    env.step(0)


def test_changed_reset_labels_are_detected(shield_class, make_shield):
    dfa = DFA([0, 1, 2], 0, [2])
    dfa.add_edge(0, 1, Atom("trigger"))
    env = make_shield(shield_class, model=np.ones((1, 1, 1)), dfa=dfa)
    env.unwrapped.labels[0] = {"trigger"}
    with pytest.raises(RuntimeError, match="Reset labels/DFA disagree"):
        env.reset()


def test_truncation_mask_supports_safe_bootstrap(shield_class, make_shield):
    env = make_shield(shield_class, ending="truncated")
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    next_q = np.array([1.0, 1000.0, 9999.0, 3.0])
    target = reward + 0.9 * next_q[info["action_mask"]].max()
    assert target == pytest.approx(0.25 + 0.9 * 3.0)
