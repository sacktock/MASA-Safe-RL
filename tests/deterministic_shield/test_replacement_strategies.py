import numpy as np
import pytest

from masa.deterministic_shield.replacement_strategies import highest_score, random_safe

MASK = np.array([True, False, False, True])


def test_random_matches_uniform_choice_from_safe_actions():
    select = random_safe(seed=123)
    reference = np.random.default_rng(123)
    expected = [int(reference.choice([0, 3])) for _ in range(128)]
    actual = [select(17, 1, MASK) for _ in range(128)]
    assert actual == expected
    assert set(actual) == {0, 3}
    assert all(type(action) is int for action in actual)


def test_random_is_reproducible_with_same_seed_and_masks():
    first, second = random_safe(seed=42), random_safe(seed=42)
    assert [first(0, 1, MASK) for _ in range(64)] == [
        second(0, 1, MASK) for _ in range(64)
    ]


def test_random_uses_current_mask_and_does_not_mutate_it():
    select = random_safe(seed=7)
    reference = np.random.default_rng(7)
    masks = [
        np.array([True, False, False, True]),
        np.array([False, True, True, False]),
        np.array([False, False, True, False]),
    ]
    for mask in masks * 8:
        before = mask.copy()
        expected = int(reference.choice(np.flatnonzero(mask)))
        assert select(0, 1, mask) == expected
        np.testing.assert_array_equal(mask, before)


def test_random_works_without_explicit_seed():
    select = random_safe()
    assert all(select(0, 1, MASK) in (0, 3) for _ in range(32))


def test_single_safe_action():
    select = random_safe(seed=5)
    mask = np.array([False, False, True, False])
    assert [select(0, 1, mask) for _ in range(8)] == [2] * 8


def test_invalid_mask_does_not_advance_random_stream():
    select, reference = random_safe(seed=4), random_safe(seed=4)
    with pytest.raises(ValueError, match="empty safe set"):
        select(0, 1, np.zeros(4, dtype=bool))
    assert [select(0, 1, MASK) for _ in range(32)] == [
        reference(0, 1, MASK) for _ in range(32)
    ]


def test_score_ties_choose_lowest_safe_index():
    mask = np.array([False, True, False, True])
    assert highest_score(lambda s: np.ones(4))(0, 2, mask) == 1


def test_scores_receive_product_observation_and_ignore_unsafe_entries():
    calls = []

    def scores(state):
        calls.append(state)
        return [1, np.nan, np.inf, 5]

    assert highest_score(scores)(17, 1, MASK) == 3
    assert calls == [17]


@pytest.mark.parametrize("scores", [
    [1, 2], [[1, 2, 3, 4]], [np.nan, 1, 2, 4], [1, 2, 3, np.inf],
])
def test_bad_scores(scores):
    with pytest.raises(ValueError):
        highest_score(lambda s: scores)(0, 1, MASK)


def test_non_callable_scores():
    with pytest.raises(TypeError):
        highest_score(np.ones(4))


@pytest.mark.parametrize("factory", [
    lambda: random_safe(seed=0), lambda: highest_score(lambda s: np.ones(4)),
])
@pytest.mark.parametrize("mask, message", [
    (np.zeros(4, dtype=bool), "empty safe set"),
    (np.ones((2, 2), dtype=bool), "one-dimensional"),
])
def test_invalid_safe_masks(factory, mask, message):
    with pytest.raises(ValueError, match=message):
        factory()(0, 1, mask)
