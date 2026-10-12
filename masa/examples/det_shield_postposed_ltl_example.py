"""Run: python -m masa.examples.postposed_ltl_example --strategy all --seed 0"""
import argparse
import numpy as np

from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.common.labelled_env import LabelledEnv
from masa.deterministic_shield import PostposedLTLShield, random_safe, highest_score
from masa.envs.tabular.colour_bomb_grid_world import ColourBombGridWorld, label_fn
from masa.examples.colour_bomb_grid_world.property_2 import make_dfa


def run(strategy: str, seed: int = 0):
    replacements = {
        "first": None,
        # Independently seeded: env.reset(seed=...) does not reset this RNG.
        "random": random_safe(seed=seed),
        # Demonstration priorities favor staying. A learned policy can instead
        # return its current Q-values/logits for the given product observation.
        "highest_score": highest_score(lambda obs: np.array([0, 0, 0, 0, 1])),
    }
    base = ColourBombGridWorld(slip_prob=0.0)
    env = PostposedLTLShield(
        LTLSafetyEnv(LabelledEnv(base, label_fn), dfa=make_dfa(), obs_type="discrete"),
        replacement=replacements[strategy],
    )
    try:
        obs, _ = env.reset(seed=seed)
        print(f"\nReplacement: {strategy}")
        for _ in range(20):
            proposed = 1  # Always try moving right, even immediately after a bomb.
            old_state = obs % base.observation_space.n
            obs, _, terminated, truncated, info = env.step(proposed)
            print(f"state={old_state}, proposed={proposed}, "
                  f"executed={info['shield_executed_action']}, "
                  f"intervened={info['shield_intervened']}, "
                  f"next={obs % base.observation_space.n}")
            if terminated or truncated:
                break
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", default="all",
        choices=("all", "first", "random", "highest_score"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in (("first", "random", "highest_score")
                 if args.strategy == "all" else (args.strategy,)):
        run(name, seed=args.seed)


if __name__ == "__main__":
    main()
