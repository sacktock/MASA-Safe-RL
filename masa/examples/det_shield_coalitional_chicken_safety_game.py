"""Helpers used by the coalition-shielding Chicken tutorials and tests."""
from __future__ import annotations

from masa.common.ltl import Atom, DFA
from masa.common.multi_agent import LabelledParallelEnv
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
