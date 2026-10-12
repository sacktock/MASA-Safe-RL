"""Compare IQL, IQL-Lambda, and IQL-Lagrangian on repeated Chicken.

Examples::

    masa example cmg_iql --algorithm all --episodes 1000 --seed 0
    python -m masa.examples.cmg_iql --algorithm iql-lagrangian --episodes 1000

The example intentionally keeps the policies independent: each agent owns a
separate MASA ``QL`` learner.  The coordinator only assembles simultaneous
PettingZoo actions and distributes the resulting per-agent transitions.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import numpy as np
from pettingzoo import ParallelEnv

from masa.algorithms.tabular.multi_agent import IQL, IQLLambda, IQLLagrangian
from masa.common.constraints.multi_agent.cmg import Budget
from masa.common.utils import make_marl_env

ALGORITHM_NAMES = ("iql", "iql-lambda", "iql-lagrangian")


def binary_state(observation: Any) -> int:
    """Encode a nonempty binary vector as a little-endian integer."""
    bits = np.asarray(observation).reshape(-1)
    if bits.size == 0 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Expected a nonempty binary observation.")
    return int(sum(int(bit) << position for position, bit in enumerate(bits)))


def make_chicken_cmg(horizon: int = 20) -> ParallelEnv:
    """Build the Chicken CMG used by the multi-agent tutorial."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    return make_marl_env(
        "ChickenMatrix",
        "CMG",
        env_kwargs={"max_moves": int(horizon)},
        constraint_kwargs={
            "budgets": [
                Budget(
                    amount=1.0,
                    agents=("player_0",),
                    name="player_0_budget",
                ),
                Budget(
                    amount=1.0,
                    agents=("player_1",),
                    name="player_1_budget",
                ),
                Budget(
                    amount=1.5,
                    agents=("player_0", "player_1"),
                    name="shared",
                ),
            ]
        },
    )


def make_learner(
    algorithm: str,
    env: ParallelEnv,
    *,
    episodes: int,
    horizon: int,
    seed: int,
    cost_lambda: float = 1.0,
    lambda_lr: float = 0.02,
    dual_update_every: int = 10,
):
    """Construct one of the three tutorial learners with matching QL settings."""
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHM_NAMES:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; choose one of {ALGORITHM_NAMES}."
        )

    ql_kwargs = {
        "alpha": 0.1,
        "gamma": 1.0,
        "exploration": "epsilon_greedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_decay_frames": max(1, int(0.8 * episodes * horizon)),
    }
    common = {
        "env": env,
        "horizon": horizon,
        "n_states": 32,
        "encode": binary_state,
        "seed": seed,
        "ql_kwargs": ql_kwargs,
    }

    if algorithm == "iql":
        return IQL(**common)
    if algorithm == "iql-lambda":
        return IQLLambda(
            **common,
            cost_lambda={"shared": cost_lambda},
        )
    return IQLLagrangian(
        **common,
        cost_lambda={"shared": 0.0},
        lambda_lr=lambda_lr,
        dual_update_every=dual_update_every,
    )


def run_variant(
    algorithm: str,
    *,
    episodes: int = 1000,
    horizon: int = 20,
    eval_episodes: int = 100,
    seed: int = 0,
    cost_lambda: float = 1.0,
    lambda_lr: float = 0.02,
    dual_update_every: int = 10,
) -> dict[str, Any]:
    """Train and greedily evaluate one variant on a fresh CMG."""
    if min(episodes, horizon, eval_episodes) < 1:
        raise ValueError("episodes, horizon, and eval_episodes must be positive.")

    env = make_chicken_cmg(horizon)
    try:
        learner = make_learner(
            algorithm,
            env,
            episodes=episodes,
            horizon=horizon,
            seed=seed,
            cost_lambda=cost_lambda,
            lambda_lr=lambda_lr,
            dual_update_every=dual_update_every,
        )
        return {
            "train": learner.train(episodes),
            "eval": learner.evaluate(
                eval_episodes,
                seed=seed + 10_000,
                deterministic=True,
            ),
            "multipliers": learner.lambdas.copy(),
            "q_tables": {
                agent: table.copy() for agent, table in learner.q_tables.items()
            },
        }
    finally:
        env.close()


def run_comparison(
    *,
    episodes: int = 1000,
    horizon: int = 20,
    eval_episodes: int = 100,
    seed: int = 0,
    cost_lambda: float = 1.0,
    lambda_lr: float = 0.02,
    dual_update_every: int = 10,
) -> dict[str, dict[str, Any]]:
    """Run all three variants independently with the same seed and horizon."""
    return {
        algorithm: run_variant(
            algorithm,
            episodes=episodes,
            horizon=horizon,
            eval_episodes=eval_episodes,
            seed=seed,
            cost_lambda=cost_lambda,
            lambda_lr=lambda_lr,
            dual_update_every=dual_update_every,
        )
        for algorithm in ALGORITHM_NAMES
    }


def _mean(rows: list[Mapping[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def print_summary(results: Mapping[str, Mapping[str, Any]]) -> None:
    """Print compact greedy-evaluation metrics."""
    print("Greedy evaluation; returns and costs are per-episode sums.")
    for name, result in results.items():
        rows = result["eval"]
        print(
            f"{name:16s} "
            f"return=({_mean(rows, 'return/player_0'):.2f}, "
            f"{_mean(rows, 'return/player_1'):.2f}) "
            f"shared_cost={_mean(rows, 'shared_cum_cost'):.3f} "
            f"shared_satisfied={_mean(rows, 'shared_satisfied'):.1%} "
            f"lambda={result['multipliers']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=("all", *ALGORITHM_NAMES),
        default="all",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost-lambda", type=float, default=1.0)
    parser.add_argument("--lambda-lr", type=float, default=0.02)
    parser.add_argument("--dual-update-every", type=int, default=10)
    args = parser.parse_args()

    if min(args.episodes, args.horizon, args.eval_episodes) < 1:
        parser.error("episodes, horizon, and eval-episodes must be positive")
    if args.cost_lambda < 0.0:
        parser.error("cost-lambda must be nonnegative")
    if args.lambda_lr <= 0.0:
        parser.error("lambda-lr must be positive")
    if args.dual_update_every < 1:
        parser.error("dual-update-every must be positive")

    kwargs = {
        "episodes": args.episodes,
        "horizon": args.horizon,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "cost_lambda": args.cost_lambda,
        "lambda_lr": args.lambda_lr,
        "dual_update_every": args.dual_update_every,
    }
    if args.algorithm == "all":
        results = run_comparison(**kwargs)
    else:
        results = {args.algorithm: run_variant(args.algorithm, **kwargs)}
    print_summary(results)


if __name__ == "__main__":
    main()
