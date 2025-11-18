from __future__ import annotations
from masa.common.ltl import *

dfa = DFA([0,1,2,3], 0, [3])
dfa.add_edge(0, 1, Atom('bomb'))
dfa.add_edge(1, 2, Atom('bomb'))
dfa.add_edge(1, 3, Neg(Atom('bomb')))
dfa.add_edge(2, 0, Neg(Atom('bomb')))