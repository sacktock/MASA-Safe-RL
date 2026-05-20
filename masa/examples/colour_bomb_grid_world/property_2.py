from __future__ import annotations
from masa.common.ltl import *

def make_dfa() -> DFA:
    """Builds a DFA for requiring one extra bomb step to diffuse a bomb.

    The automaton encodes the property that, after entering a ``bomb`` state,
    leaving ``bomb`` immediately is unsafe. Equivalently, an agent that hits a
    bomb must remain on ``bomb`` for one additional transition before returning
    to safety.

    .. math::

        G\\left((\\neg \\texttt{bomb} \\land X\\,\\texttt{bomb})
        \\implies X\\,\\texttt{bomb}\\right)

    The accepting state ``3`` represents a violation: the agent hits ``bomb``
    and then immediately observes ``\\neg bomb`` instead of staying on
    ``bomb`` for the required extra step.

    Returns:
        DFA: A :class:`DFA` whose accepting state indicates violation of the
        bomb-diffusion requirement.
    """
    dfa = DFA([0,1,2,3], 0, [3])
    dfa.add_edge(0, 1, Atom('bomb'))
    dfa.add_edge(1, 2, Atom('bomb'))
    dfa.add_edge(1, 3, Neg(Atom('bomb')))
    dfa.add_edge(2, 0, Neg(Atom('bomb')))
    return dfa