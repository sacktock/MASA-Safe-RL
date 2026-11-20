"""Helpers used by the coalition-shielding Chicken tutorials and tests.

Run: python -m masa.examples.det_shield_coalitional_chicken_safety_game
"""

from __future__ import annotations

import numpy as np

from masa.common.ltl import Atom, DFA
from masa.common.multi_agent import LabelledParallelEnv
from masa.deterministic_shield import CoalitionLTLShield
from masa.envs.multiagent.matrix.chicken import ChickenMatrix, label_fn

PLAYER_0 = "player_0"
PLAYER_1 = "player_1"
AGENTS = (PLAYER_0, PLAYER_1)
SWERVE = 0
STRAIGHT = 1


def make_labelled_chicken_env(**kwargs) -> LabelledParallelEnv:
    """Create model-enabled repeated Chicken with runtime proposition labels."""
    return LabelledParallelEnv(ChickenMatrix(**kwargs), label_fn)


def make_never_crash_dfa() -> DFA:
    """Return the bad-prefix monitor for ``G(not crash)``."""
    dfa = DFA([0, 1], initial=0, accepting=[1])
    dfa.add_edge(0, 1, Atom("crash"))
    return dfa


__all__ = [
    "AGENTS",
    "PLAYER_0",
    "PLAYER_1",
    "STRAIGHT",
    "SWERVE",
    "make_labelled_chicken_env",
    "make_never_crash_dfa",
]


def run_random(env, episodes: int = 10, seed: int = 0) -> tuple[int, int, int]:
    """Play uniformly random joint actions; return (crashes, interventions, steps)."""
    rng = np.random.default_rng(seed)
    crashes = interventions = steps = 0
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        while env.agents:
            actions = {agent: int(rng.integers(2)) for agent in AGENTS}
            _, _, _, _, infos = env.step(actions)
            crashes += "crash" in infos[PLAYER_0]["labels"]
            interventions += infos[PLAYER_0].get("shield_intervened", False)
            steps += 1
    env.close()
    return crashes, interventions, steps


def main():
    unshielded = make_labelled_chicken_env(max_moves=50)
    # player_0 is shielded; player_1 stays an unrestricted (random) adversary.
    shielded = CoalitionLTLShield(
        make_labelled_chicken_env(max_moves=50),
        coalition=(PLAYER_0,),
        dfa=make_never_crash_dfa(),
        mode="postposed",
    )
    for name, env in (("unshielded", unshielded), ("shielded", shielded)):
        crashes, interventions, steps = run_random(env)
        print(
            f"{name:>10}: crashes={crashes}/{steps} steps, interventions={interventions}"
        )


if __name__ == "__main__":
    main()
