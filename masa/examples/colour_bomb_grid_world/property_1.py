from __future__ import annotations
from masa.common.ltl import *

dfa = DFA([0, 1], 0, [1])
dfa.add_edge(0, 1, Atom('bomb'))

