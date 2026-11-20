from __future__ import annotations

import pytest

from masa.common import registry
from masa.common.constraints.multi_agent.cmg import Budget
from masa.common.utils import accepts_kwarg, make_env, make_marl_env
from masa.envs.multiagent.matrix.chicken import cost_fn as chicken_cost_fn


@pytest.mark.parametrize("constraint", sorted(registry.CONSTRAINT_REGISTRY.keys()))
def test_constraint_wrappers_reject_unknown_kwargs(constraint):
    with pytest.raises(TypeError, match="not_a_kwarg"):
        make_env(
            "ColourGridWorld",
            constraint,
            10,
            constraint_kwargs={"not_a_kwarg": 1.0},
        )


def test_cmdp_rejects_misspelled_budget():
    with pytest.raises(TypeError, match="cost_budget"):
        make_env(
            "ColourGridWorld",
            "CMDP",
            10,
            constraint_kwargs={"cost_budget": 0.05},
        )


@pytest.mark.parametrize(
    "constraint, kwargs",
    [
        ("CMDP", {"alpha": 0.01}),
        ("PCTL", {"budget": 1.0}),
        ("REACH_AVOID", {"budget": 1.0}),
        ("LTL_SAFETY", {"alpha": 0.01}),
    ],
)
def test_constraint_wrappers_reject_kwargs_for_other_constraints(constraint, kwargs):
    with pytest.raises(TypeError, match=next(iter(kwargs))):
        make_env("ColourGridWorld", constraint, 10, constraint_kwargs=kwargs)


def test_cmg_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="budget"):
        make_marl_env(
            "ChickenMatrix",
            "CMG",
            constraint_kwargs={
                "budgets": [
                    Budget(amount=1.0, agents=("player_0", "player_1"), name="shared")
                ],
                "budget": 1.0,
            },
        )


@pytest.mark.parametrize("constraint", ["REACH_AVOID", "LTL_SAFETY"])
def test_make_env_builds_constraints_without_cost_fn_on_env_with_cost_fn(constraint):
    env = make_env("FrozenLake", constraint, 10)
    env.reset(seed=0)
    env.close()


def test_make_marl_env_forwards_env_cost_fn_when_constraint_accepts_it():
    env = make_marl_env(
        "ChickenMatrix",
        "CMG",
        constraint_kwargs={
            "budgets": [
                Budget(amount=1.0, agents=("player_0", "player_1"), name="shared")
            ]
        },
    )

    assert env.cost_fn is chicken_cost_fn


def test_make_marl_env_skips_env_cost_fn_when_constraint_does_not_accept_it(
    monkeypatch,
):
    calls = []

    def no_cost_constraint(env, budgets):
        calls.append(budgets)
        return env

    monkeypatch.setitem(
        registry.MARL_CONSTRAINT_REGISTRY._items, "NO_COST", no_cost_constraint
    )

    make_marl_env("ChickenMatrix", "NO_COST", constraint_kwargs={"budgets": []})

    assert calls == [[]]


def test_accepts_kwarg_inspects_constructor_signature():
    def keyword_params(a, *, b):
        pass

    def positional_only(a, /):
        pass

    def var_keyword(**kwargs):
        pass

    assert accepts_kwarg(keyword_params, "a")
    assert accepts_kwarg(keyword_params, "b")
    assert not accepts_kwarg(keyword_params, "c")
    assert not accepts_kwarg(positional_only, "a")
    assert accepts_kwarg(var_keyword, "anything")
