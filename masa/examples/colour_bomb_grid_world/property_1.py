from __future__ import annotations
from masa.common.ltl import *

def make_dfa() -> DFA:
    """Builds a DFA for the safety property “never hit a bomb”.

    The automaton represents the invariant

    .. math::

        G\\,\\neg \\texttt{bomb}

    State ``0`` is the safe initial state. Observing ``bomb`` moves the
    automaton to accepting sink/error state ``1``.

    Returns:
        DFA: A :class:`DFA` whose accepting state indicates violation of the
        invariant.
    """
    dfa = DFA([0, 1], 0, [1])
    dfa.add_edge(0, 1, Atom('bomb'))
    return dfa

