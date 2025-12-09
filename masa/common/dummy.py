from __future__ import annotations
from masa.common.ltl import DFA

label_fn = lambda obs: set()
cost_fn = lambda labels: 0.0
make_dfa = lambda: DFA([0], 0, [])