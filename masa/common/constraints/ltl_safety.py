from __future__ import annotations
from typing import Any, Dict
from .base import ConstraintEnv
from model_checking.dfa import DFA, dfa_to_costfn

class LTLSafetyEnv(ConstraintEnv):

    def __init__(self, dfa: DFA):
        super().__init__(env, cost_fn=dfa_to_costfn(dfa))

    def _reset(self):
        self.safe = True
        self.cost_fn.reset()

    def _update(self, info):
        cost = self.cost_fn(info)
        unsafe = float(cost >= 0.5)
        self.safe = self.safe and (not unsafe)

    def get_automaton_state(self):
        return self.cost_fn.automaton_state

    def satisfied(self) -> bool:
        return self.safe

    def episode_metric(self) -> Dict[str, float]:
        return {"satisfied": float(self.satisfied()), "automaton_state": self.get_automaton_state()}
    

        