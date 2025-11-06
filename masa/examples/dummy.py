from __future__ import annotations
from typing import Iterable, Any
from masa.common.ltl import DFA

def label_fn(obs: Any) -> Iterable[str]:
    return {}

def cost_fn(labels: Iterable[str]) -> float:
    return 0.0

def dfa() -> DFA:
    return DFA([0], 0, [])