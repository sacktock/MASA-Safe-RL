from __future__ import annotations
from masa.common.ltl import *

def make_dfa() -> DFA:
    """Builds a DFA for post-bomb medic recovery.

    The automaton encodes the bounded temporal requirement that whenever the
    agent hits a ``bomb`` state, it must subsequently reach a ``medic`` state
    within ten steps and remain in ``medic`` for two consecutive steps.

    .. math::

        G\\left(\\texttt{bomb} \\implies
        X\\,F_{\\leq 10}\\,G_{\\leq 2}\\,\\texttt{medic}\\right)

    State ``0`` is the nominal safe state. States ``1`` through ``10`` count
    how many steps have elapsed since the bomb was observed while waiting for
    ``medic``. States ``11`` through ``20`` represent having observed the first
    required ``medic`` step. State ``21`` is the accepting violation state,
    reached if the agent fails to satisfy the bounded recovery condition.

    Returns:
        DFA: A :class:`DFA` whose accepting state indicates violation of the
        bounded medic-recovery requirement.
    """
    dfa = DFA([i for i in range(22)], 0, [21])
    dfa.add_edge(0, 1, Atom('bomb'))

    dfa.add_edge(1, 2, Neg(Atom('medic')))
    dfa.add_edge(1, 11, Atom('medic'))

    dfa.add_edge(2, 3, Neg(Atom('medic')))
    dfa.add_edge(2, 12, Atom('medic'))

    dfa.add_edge(3, 4, Neg(Atom('medic')))
    dfa.add_edge(3, 13, Atom('medic'))

    dfa.add_edge(4, 5, Neg(Atom('medic')))
    dfa.add_edge(4, 14, Atom('medic'))

    dfa.add_edge(5, 6, Neg(Atom('medic')))
    dfa.add_edge(5, 15, Atom('medic'))

    dfa.add_edge(6, 7, Neg(Atom('medic')))
    dfa.add_edge(6, 16, Atom('medic'))

    dfa.add_edge(7, 8, Neg(Atom('medic')))
    dfa.add_edge(7, 17, Atom('medic'))

    dfa.add_edge(8, 9, Neg(Atom('medic')))
    dfa.add_edge(8, 18, Atom('medic'))

    dfa.add_edge(9, 10, Neg(Atom('medic')))
    dfa.add_edge(9, 19, Atom('medic'))

    dfa.add_edge(10, 21, Neg(Atom('medic')))
    dfa.add_edge(10, 20, Atom('medic'))

    dfa.add_edge(11, 0, Atom('medic'))
    dfa.add_edge(11, 3, Neg(Atom('medic')))

    dfa.add_edge(12, 0, Atom('medic'))
    dfa.add_edge(12, 4, Neg(Atom('medic')))

    dfa.add_edge(13, 0, Atom('medic'))
    dfa.add_edge(13, 5, Neg(Atom('medic')))

    dfa.add_edge(14, 0, Atom('medic'))
    dfa.add_edge(14, 6, Neg(Atom('medic')))

    dfa.add_edge(15, 0, Atom('medic'))
    dfa.add_edge(15, 7, Neg(Atom('medic')))

    dfa.add_edge(16, 0, Atom('medic'))
    dfa.add_edge(16, 8, Neg(Atom('medic')))

    dfa.add_edge(17, 0, Atom('medic'))
    dfa.add_edge(17, 9, Neg(Atom('medic')))

    dfa.add_edge(18, 0, Atom('medic'))
    dfa.add_edge(18, 10, Neg(Atom('medic')))

    dfa.add_edge(19, 0, Atom('medic'))
    dfa.add_edge(19, 21, Neg(Atom('medic')))

    dfa.add_edge(20, 0, Atom('medic'))
    dfa.add_edge(20, 21, Neg(Atom('medic')))

    return dfa