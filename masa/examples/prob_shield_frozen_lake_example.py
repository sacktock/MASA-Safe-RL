from __future__ import annotations

import numpy as np

from masa.common.utils import make_env
from masa.prob_shield.prob_shield_wrapper_v1 import ProbShieldWrapperDisc


def build_env():
    return make_env(
        "FrozenLake",
        "PCTL",
        100,
        env_kwargs={"is_slippery": False},
        constraint_kwargs={"alpha": 0.01},
    )


def main():
    env = ProbShieldWrapperDisc(
        build_env(),
        init_safety_bound=1e-12,
        theta=1e-12,
        max_vi_steps=2_000,
        granularity=10,
    )

    obs, info = env.reset(seed=0)
    print("Initial observation:", obs)
    print("Initial labels:", info["labels"])
    print("Initial safety lower bound:", env.safety_lb[env._current_obs])
    print("Max successors:", env.max_successors)

    action = np.zeros(2 + env.max_successors, dtype=np.int64)
    action[0] = 2  # right
    action[1] = 1  # down

    safe_actions, bounds, proj_penalty, margin_penalty = env._project_act(*env._parse_act(action))
    print("Projected action distribution:", safe_actions)
    print("Projected successor bounds:", bounds)
    print("Projection penalty:", proj_penalty)
    print("Margin penalty:", margin_penalty)

    next_obs, reward, terminated, truncated, info = env.step(action)
    print("Next observation:", next_obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Constraint info:", info["constraint"])

    env.close()


if __name__ == "__main__":
    main()
