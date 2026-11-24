from __future__ import annotations

import numpy as np
import pytest

import masa.common.pctl as pctl

N = 6
P_UP = 0.4


def gamblers_ruin_kernel() -> np.ndarray:
    """Dense (S, S) kernel, column s = next-state distribution from s; 0 and N absorbing."""
    S = N + 1
    m = np.zeros((S, S))
    m[0, 0] = 1.0
    m[N, N] = 1.0
    for s in range(1, N):
        m[s + 1, s] = P_UP
        m[s - 1, s] = 1.0 - P_UP
    return m


def gamblers_ruin_exact() -> np.ndarray:
    r = (1.0 - P_UP) / P_UP
    return np.array([(1.0 - r**s) / (1.0 - r**N) for s in range(N + 1)])


def dense_to_compact(m: np.ndarray):
    S = m.shape[1]
    K = int(np.count_nonzero(m, axis=0).max())
    succ = -np.ones((K, S), dtype=np.int64)
    p = np.zeros((K, S))
    for s in range(S):
        nz = np.flatnonzero(m[:, s])
        succ[: nz.size, s] = nz
        p[: nz.size, s] = m[nz, s]
    return succ, p


def goal_label_fn(goal_states):
    goal_states = set(goal_states)
    return lambda s: {"goal"} if s in goal_states else set()


@pytest.mark.parametrize("method", pctl.LINEAR_SOLVERS)
def test_gamblers_ruin_matches_closed_form(method):
    m = gamblers_ruin_kernel()
    sat2 = np.zeros(N + 1)
    sat2[N] = 1.0
    x = pctl.solve_until_probabilities(m, np.ones(N + 1), sat2, method=method)
    np.testing.assert_allclose(x, gamblers_ruin_exact(), atol=1e-12)


@pytest.mark.parametrize("method", pctl.LINEAR_SOLVERS)
def test_dense_and_compact_kernels_agree(method):
    m = gamblers_ruin_kernel()
    sat2 = np.zeros(N + 1)
    sat2[N] = 1.0
    x_dense = pctl.solve_until_probabilities(m, np.ones(N + 1), sat2, method=method)
    x_compact = pctl.solve_until_probabilities(
        dense_to_compact(m), np.ones(N + 1), sat2, method=method
    )
    np.testing.assert_allclose(x_dense, x_compact, atol=1e-12)


def test_bounded_eventually_converges_to_unbounded():
    m = gamblers_ruin_kernel()
    vec_label_fn = np.zeros((1, N + 1))
    vec_label_fn[0, N] = 1.0
    atom_dict = {"goal": 0}
    goal = pctl.Atom("goal")

    unbounded = pctl.UnboundedEventually(0.5, goal)._prob_vector(
        m, vec_label_fn, atom_dict
    )
    bounded = pctl.Eventually(0.5, 2000, goal)._prob_seq(m, vec_label_fn, atom_dict)[-1]
    np.testing.assert_allclose(unbounded, bounded, atol=1e-10)
    np.testing.assert_allclose(unbounded, gamblers_ruin_exact(), atol=1e-12)


@pytest.mark.parametrize("method", pctl.LINEAR_SOLVERS)
def test_prob0_handles_closed_class_that_misses_target(method):
    # 0 -> {1, 2}; {1, 3} is a closed class that never reaches the target 2.
    m = np.zeros((4, 4))
    m[1, 0], m[2, 0] = 0.3, 0.7
    m[3, 1] = 1.0
    m[1, 3] = 1.0
    m[2, 2] = 1.0
    sat2 = np.array([0.0, 0.0, 1.0, 0.0])
    x = pctl.solve_until_probabilities(m, np.ones(4), sat2, method=method)
    np.testing.assert_allclose(x, [0.7, 0.0, 1.0, 0.0], atol=1e-12)


def test_unbounded_until_respects_continuation_condition():
    m = gamblers_ruin_kernel()
    sat1 = np.ones(N + 1)
    sat1[3] = 0.0  # paths may not pass through state 3
    sat2 = np.zeros(N + 1)
    sat2[N] = 1.0
    x = pctl.solve_until_probabilities(m, sat1, sat2)
    assert np.all(x[:4] == 0.0)
    assert x[N] == 1.0
    assert 0.0 < x[4] < 1.0


def test_unknown_solver_raises():
    with pytest.raises(ValueError):
        pctl.solve_until_probabilities(np.eye(2), np.ones(2), np.zeros(2), method="qr")


def test_exact_checker_state_action_with_unbounded_formula():
    # 2-action MDP on the gambler's-ruin states: action 0 = gamble, action 1 = stay put.
    m = gamblers_ruin_kernel()
    S = N + 1
    tm = np.zeros((S, S, 2))
    tm[:, :, 0] = m
    tm[:, :, 1] = np.eye(S)

    formula = pctl.UnboundedEventually(0.5, pctl.Atom("goal"))
    policy = np.zeros((S, 2))
    policy[:, 0] = 1.0

    for solver in pctl.LINEAR_SOLVERS:
        checker = pctl.ExactModelChecker(
            formula,
            goal_label_fn({N}),
            ["goal"],
            transition_matrix=tm,
            linear_solver=solver,
        )
        assert formula._inner.method == solver
        Q = checker.check_state_action(None, policy)
        V = gamblers_ruin_exact()
        np.testing.assert_allclose(Q[:, 0], m.T @ V, atol=1e-12)
        np.testing.assert_allclose(Q[:, 1], V, atol=1e-12)
        np.testing.assert_array_equal(
            checker.check_state(None, policy), (V >= 0.5).astype(float)
        )


def test_unbounded_always_is_dual_of_eventually():
    m = gamblers_ruin_kernel()
    vec_label_fn = np.zeros((1, N + 1))
    vec_label_fn[0, 0] = 1.0  # "unsafe" = ruin
    atom_dict = {"unsafe": 0}
    safe = pctl.Neg(pctl.Atom("unsafe"))

    always = pctl.UnboundedAlways(0.5, safe)
    x = always._prob_vector(m, vec_label_fn, atom_dict)
    np.testing.assert_allclose(x, gamblers_ruin_exact(), atol=1e-12)
    np.testing.assert_allclose(always._prob_seq(m, vec_label_fn, atom_dict)[0], x)


def test_statistical_checker_rejects_unbounded_formulas():
    tm = np.zeros((2, 2, 1))
    tm[0, 0, 0] = tm[1, 1, 0] = 1.0
    with pytest.raises(NotImplementedError):
        pctl.StatisticalModelChecker(
            pctl.UnboundedEventually(0.5, pctl.Atom("goal")),
            goal_label_fn({1}),
            ["goal"],
            transition_matrix=tm,
        )


def _frozen_lake_env():
    from masa.common.utils import make_env
    from masa.envs.tabular.frozen_lake import cost_fn

    return make_env(
        "FrozenLake",
        "PCTL",
        100,
        env_kwargs={"is_slippery": True},
        constraint_kwargs={"alpha": 0.01, "cost_fn": cost_fn},
    )


@pytest.mark.parametrize("algo", ["RECREG", "RECREG_BUDGET"])
def test_recreg_model_checking_flag_validation(algo):
    from masa.algorithms import tabular

    cls = getattr(tabular, algo)
    with pytest.raises(AssertionError, match="model_based"):
        cls(_frozen_lake_env(), mode="model_based", model_checking="exact")
    with pytest.raises(AssertionError, match="statistical"):
        cls(_frozen_lake_env(), mode="exact", model_checking="statistical")

    assert (
        getattr(tabular, f"{algo}_MODEL_BASED")(_frozen_lake_env()).model_checking
        == "finite"
    )


@pytest.mark.parametrize("algo", ["RECREG_EXACT", "RECREG_BUDGET_EXACT"])
def test_recreg_exact_mode_unbounded_vs_finite(algo):
    from masa.algorithms import tabular

    cls = getattr(tabular, algo)
    exact = cls(_frozen_lake_env(), horizon=5)
    assert exact.model_checking == "exact"
    assert isinstance(exact.model_checker.formula, pctl.UnboundedAlways)

    finite_short = cls(_frozen_lake_env(), model_checking="finite", horizon=5)
    finite_long = cls(_frozen_lake_env(), model_checking="finite", horizon=500)
    assert isinstance(finite_short.model_checker.formula, pctl.Always)

    np.testing.assert_allclose(exact.S, finite_long.S, atol=1e-10)
    assert np.all(finite_short.S <= exact.S + 1e-12)
    assert np.any(finite_short.S < exact.S - 1e-6)
