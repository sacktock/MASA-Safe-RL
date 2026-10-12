import numpy as np
import pytest

from masa.deterministic_shield import (
    DeterministicLTLShield, PostposedLTLShield, highest_score, random_safe,
)
from masa.deterministic_shield.deterministic_shield import (
    DeterministicLTLShield as LegacyShield,
)


def test_legacy_imports():
    assert DeterministicLTLShield is LegacyShield is PostposedLTLShield


@pytest.mark.parametrize("sparse", [False, True])
def test_default_is_first_safe_and_preserves_safe_proposals(make_shield, sparse):
    env = make_shield(PostposedLTLShield, sparse=sparse)
    env.reset(seed=0)
    for proposed, expected in ((1, 0), (2, 0), (3, 3), (0, 0), (1, 0)):
        _, reward, term, trunc, info = env.step(proposed)
        assert info["shield_executed_action"] == expected
        assert info["shield_intervened"] == (proposed != expected)
        assert (reward, term, trunc) == (0.25, False, False)


@pytest.mark.parametrize("rule", ["score", "custom"])
def test_replacement_strategies(make_shield, rule):
    replacement = {
        # The biggest scores are unsafe: they must be ignored.
        "score": highest_score(lambda state: np.array([1, 1000, 9999, 7])),
        "custom": lambda state, proposed, mask: np.int64(3),
    }[rule]
    env = make_shield(PostposedLTLShield, replacement=replacement)
    env.reset(seed=0)
    for _ in range(5):
        info = env.step(1)[4]
        assert info["shield_executed_action"] == 3
    assert env.unwrapped.executed == [3] * 5


def test_callback_only_runs_for_unsafe_proposals(make_shield):
    calls = []
    def replacement(state, proposed, mask):
        calls.append((state, proposed, mask.copy()))
        return 3
    env = make_shield(PostposedLTLShield, replacement=replacement)
    obs, _ = env.reset()
    env.step(0)
    env.step(3)
    assert not calls
    env.step(1)
    assert len(calls) == 1
    assert calls[0][:2] == (obs, 1)
    np.testing.assert_array_equal(calls[0][2], [True, False, False, True])


@pytest.mark.parametrize("result", [2, -1, 4, 0.5, None])
def test_invalid_callback_result_never_reaches_environment(make_shield, result):
    env = make_shield(
        PostposedLTLShield, replacement=lambda state, proposed, mask: result
    )
    env.reset()
    with pytest.raises(ValueError):
        env.step(1)
    assert env.unwrapped.executed == []
    env.step(3)  # Invalid replacement did not invalidate an unchanged episode.
    assert env.unwrapped.executed == [3]


def test_callback_cannot_change_internal_mask(make_shield):
    def replacement(state, proposed, mask):
        mask[:] = True
        return 2  # Still unsafe, regardless of the mutated copy.
    env = make_shield(PostposedLTLShield, replacement=replacement)
    env.reset()
    with pytest.raises(ValueError, match="not safe"):
        env.step(1)
    assert env.unwrapped.executed == []
    np.testing.assert_array_equal(env.action_masks(), [True, False, False, True])


def test_callback_exception_does_not_advance_episode(make_shield):
    def replacement(state, proposed, mask):
        raise LookupError("policy unavailable")
    env = make_shield(PostposedLTLShield, replacement=replacement)
    env.reset()
    with pytest.raises(LookupError, match="policy unavailable"):
        env.step(1)
    assert env.unwrapped.executed == []
    env.step(0)


def test_bad_replacement_configuration(make_shield):
    with pytest.raises(TypeError, match="callable"):
        make_shield(PostposedLTLShield, replacement="random")


@pytest.mark.parametrize("sparse", [False, True])
def test_random_replacements_are_safe_and_seeded(make_shield, sparse):
    env = make_shield(
        PostposedLTLShield, sparse=sparse, replacement=random_safe(seed=123)
    )
    reference = np.random.default_rng(123)
    try:
        env.reset(seed=0)
        expected = [int(reference.choice([0, 3])) for _ in range(64)]
        for action in expected:
            info = env.step(1)[4]
            assert info["shield_proposed_action"] == 1
            assert info["shield_executed_action"] == action
            assert info["shield_intervened"]
            assert env._constraint.satisfied()
        assert env.unwrapped.executed == expected
        assert set(expected) == {0, 3}
    finally:
        env.close()


def test_safe_proposals_do_not_consume_replacement_randomness(make_shield):
    env = make_shield(PostposedLTLShield, replacement=random_safe(seed=23))
    reference = np.random.default_rng(23)
    try:
        env.reset()
        for _ in range(32):
            for safe in (0, 3):
                info = env.step(safe)[4]
                assert info["shield_executed_action"] == safe
                assert not info["shield_intervened"]
            assert env.step(1)[4]["shield_executed_action"] == int(
                reference.choice([0, 3])
            )
    finally:
        env.close()


def test_reset_does_not_reseed_the_replacement_rng(make_shield):
    env = make_shield(PostposedLTLShield, replacement=random_safe(seed=99))
    reference = np.random.default_rng(99)
    try:
        # Even repeated identical environment seeds must not rewind the selector.
        for _ in range(4):
            env.reset(seed=1)
            for _ in range(8):
                assert env.step(1)[4]["shield_executed_action"] == int(
                    reference.choice([0, 3])
                )
    finally:
        env.close()


def test_random_replacement_does_not_consume_environment_rng(make_shield):
    env = make_shield(PostposedLTLShield, replacement=random_safe(seed=9))
    control = make_shield(PostposedLTLShield)
    try:
        env.reset(seed=41)
        control.reset(seed=41)
        for _ in range(16):
            # Both safe actions self-loop; the base consumes one identical draw.
            env.step(1)
            control.step(0)
        np.testing.assert_array_equal(
            env.unwrapped.np_random.random(8),
            control.unwrapped.np_random.random(8),
        )
    finally:
        env.close()
        control.close()
