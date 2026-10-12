import numpy as np
import pytest

from masa.deterministic_shield import PreemptiveLTLShield


@pytest.mark.parametrize("sparse", [False, True])
def test_unsafe_choice_does_not_step_or_replace(make_shield, sparse):
    env = make_shield(PreemptiveLTLShield, sparse=sparse)
    obs, info = env.reset(seed=0)
    np.testing.assert_array_equal(info["action_mask"], [True, False, False, True])
    q = env._constraint.get_automaton_state()
    for action in (1, 2):
        with pytest.raises(ValueError, match="not safe"):
            env.step(action)
    assert env.unwrapped.executed == []
    assert env._constraint.get_automaton_state() == q
    assert env.unwrapped.state == obs == 0

    # A rejected action leaves the episode active; no reset is required.
    obs, reward, term, trunc, info = env.step(3)
    assert (obs, reward, term, trunc) == (0, 0.25, False, False)
    assert env.unwrapped.executed == [3]
    assert not info["shield_intervened"]
    assert info["shield_proposed_action"] == info["shield_executed_action"] == 3
    assert info["base_info"] == "preserved"


def test_masks_are_copies(make_shield):
    env = make_shield(PreemptiveLTLShield)
    _, info = env.reset()
    info["action_mask"][:] = False
    mask = env.action_masks()
    mask[:] = True
    np.testing.assert_array_equal(env.action_masks(), [True, False, False, True])
    with pytest.raises(ValueError):
        env.safe_actions[0, 0] = False
    with pytest.raises(ValueError):
        env.winning_region[0] = False


def test_mask_aware_policy_never_needs_replacement(make_shield):
    env = make_shield(PreemptiveLTLShield)
    _, info = env.reset(seed=5)
    rng = np.random.default_rng(5)
    for _ in range(30):
        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
        _, _, _, _, info = env.step(action)
        assert info["shield_executed_action"] == action
        assert not info["shield_intervened"]
    assert set(env.unwrapped.executed) <= {0, 3}
