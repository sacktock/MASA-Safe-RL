from __future__ import annotations
from masa.common.ltl import *

def make_dfa() -> DFA:
    dfa = DFA([0, 1], 0, [1])
    dfa.add_edge(0, 1, Atom('bomb'))
    return dfa

