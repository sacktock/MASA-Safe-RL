r"""
Lightweight propositional (LTL-style) formula objects and a deterministic finite
automaton (DFA) whose transitions are guarded by those formulae.

Core ideas:
  - A label is an iterable of atomic proposition names (strings).
  - A :class:`Formula` implements a satisfaction predicate :meth:`Formula.sat`.
  - A DFA transition from :math:`q` to :math:`q'` is enabled when the edge guard
    formula is satisfied by the current label set.
  - :class:`DFACostFn` wraps a :class:`DFA` as a MASA :class:`~masa.common.constraints.base.CostFn`
    where the cost is ``1.0`` upon reaching accepting states (after stepping) and
    ``0.0`` otherwise.

This module is intentionally minimal: it does not implement full LTL temporal
operators (X, U, F, G). Instead, it provides propositional guards for DFA edges,
which is sufficient for DFA-based monitoring and safety cost construction.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Iterable, List, Tuple

from masa.common.constraints.base import CostFn


class Formula:
    r"""Base class for propositional formulae over atomic proposition labels.

    A formula is evaluated against a *label set* (an iterable of strings), and
    returns whether the formula is satisfied.

    Subclasses must implement :meth:`sat`, which defines the satisfaction
    relation between the formula and a set of labels.

    Notes:
      The operators provided here are propositional (Boolean) connectives only.
      Temporal structure is represented externally via a :class:`DFA` that
      consumes a trace of label sets.
    """

    def sat(self, labels: Iterable[str]) -> bool:
        """Checks whether the formula is satisfied by the given labels.

        Args:
          labels: Iterable of atomic proposition names that hold at the current
            step.

        Returns:
          ``True`` if the formula is satisfied under the given labels.

        Raises:
          NotImplementedError: If the subclass does not implement the
            satisfaction relation.
        """
        raise NotImplementedError(
            "Propositional formula must implement a satisfaction relation"
        )


class Atom(Formula):
    r"""Atomic proposition.

    An :class:`Atom` is satisfied iff the stored atomic proposition name appears
    in the given label set.

    Attributes:
      atom: The atomic proposition name.
    """

    def __init__(self, atom: str):
        self.atom = atom

    def sat(self, labels: Iterable[str]) -> bool:
        """Evaluates whether the atom is present in ``labels``.

        Args:
          labels: Iterable of atomic proposition names.

        Returns:
          ``True`` iff :attr:`atom` is in ``labels``.
        """
        return self.atom in labels


class Truth(Formula):
    r"""Constant truth.

    Always satisfied, regardless of the labels.
    """

    def __init__(self):
        pass

    def sat(self, labels: Iterable[str]) -> bool:
        """Evaluates the truth constant.

        Args:
          labels: Unused. Included for API consistency.

        Returns:
          ``True``.
        """
        return True


class And(Formula):
    r"""Conjunction of two subformulae.

    Satisfied iff both subformulae are satisfied.

    Attributes:
      subformula_1: Left operand.
      subformula_2: Right operand.
    """

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        """Evaluates conjunction satisfaction.

        Args:
          labels: Iterable of atomic proposition names.

        Returns:
          ``True`` iff both :attr:`subformula_1` and :attr:`subformula_2` are
          satisfied by ``labels``.
        """
        return self.subformula_1.sat(labels) and self.subformula_2.sat(labels)


class Or(Formula):
    r"""Disjunction of two subformulae.

    Satisfied iff at least one subformula is satisfied.

    Attributes:
      subformula_1: Left operand.
      subformula_2: Right operand.
    """

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        """Evaluates disjunction satisfaction.

        Args:
          labels: Iterable of atomic proposition names.

        Returns:
          ``True`` iff either :attr:`subformula_1` or :attr:`subformula_2` is
          satisfied by ``labels``.
        """
        return self.subformula_1.sat(labels) or self.subformula_2.sat(labels)


class Neg(Formula):
    r"""Negation of a subformula.

    Satisfied iff the subformula is not satisfied.

    Attributes:
      subformula: The formula being negated.
    """

    def __init__(self, subformula: Formula):
        self.subformula = subformula

    def sat(self, labels: Iterable[str]) -> bool:
        """Evaluates negation satisfaction.

        Args:
          labels: Iterable of atomic proposition names.

        Returns:
          ``True`` iff :attr:`subformula` is not satisfied by ``labels``.
        """
        return not self.subformula.sat(labels)


class Implies(Formula):
    r"""Implication between two subformulae.

    ``Implies(a, b)`` is satisfied iff either ``a`` is false or ``b`` is true
    under the given labels, i.e. it is equivalent to:

    .. math::

       \neg a \lor b

    Attributes:
      subformula_1: Antecedent (premise).
      subformula_2: Consequent (conclusion).
    """

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        """Evaluates implication satisfaction.

        Args:
          labels: Iterable of atomic proposition names.

        Returns:
          ``True`` iff the implication holds under ``labels``.
        """
        return Or(Neg(self.subformula_1), self.subformula_2).sat(labels)


class DFA:
    r"""Deterministic finite automaton with propositional guards on edges.

    Each edge from a parent state to a child state is labelled with a guard
    :class:`Formula`. A transition is taken when its guard is satisfied by the
    current label set.

    Attributes:
      states: List of automaton states.
      initial: Initial automaton state.
      accepting: List of accepting (final) states.
      edges: Transition structure mapping ``parent -> {child: guard}``.
      state: Current automaton state used by :meth:`step`.

    Notes:
      The transition relation is deterministic by convention: if multiple
      outgoing guards from a state are simultaneously satisfied, the first one
      encountered in iteration order is taken. For strict determinism, ensure
      guards are mutually exclusive.
    """

    def __init__(self, states: List[int], initial: int, accepting: List[int]):
        """Creates a DFA.

        Args:
          states: List of automaton states (typically integers).
          initial: Initial state.
          accepting: Accepting (final) states.
        """
        self.states = states
        self.initial = initial
        self.accepting = accepting
        self.edges = {s: {} for s in self.states}
        self.state = self.initial

    def add_edge(self, parent: int, child: int, condition: Formula):
        """Adds a guarded transition ``parent -> child``.

        Args:
          parent: Source state.
          child: Destination state.
          condition: Guard formula enabling this transition when satisfied.

        Notes:
          This overwrites any existing edge guard between the same parent/child
          pair.
        """
        self.edges[parent][child] = condition

    def reset(self) -> int:
        """Resets the DFA to the initial state.

        Returns:
          The reset state (i.e., :attr:`initial`).
        """
        self.state = self.initial
        return self.state

    def has_edge(self, state_1: int, state_2: int) -> bool:
        """Checks whether there is an edge ``state_1 -> state_2``.

        Args:
          state_1: Source state.
          state_2: Destination state.

        Returns:
          ``True`` iff an explicit edge from ``state_1`` to ``state_2`` exists.
        """
        try:
            _ = self.edges[state_1][state_2]
            return True
        except KeyError:
            return False

    def check(self, trace: Iterable[Iterable[str]]) -> bool:
        """Checks whether a trace is accepted by the DFA.

        Args:
          trace: Sequence of label sets, one per time step.

        Returns:
          ``True`` iff the state reached after consuming the full trace is in
          :attr:`accepting`.
        """
        state = self.initial
        for labels in trace:
            state = self.transition(state, labels)
        return state in self.accepting

    def transition(self, state: int, labels: Iterable[str]) -> int:
        """Computes the next automaton state given the current state and labels.

        Args:
          state: Current DFA state.
          labels: Iterable of atomic proposition names holding at the current
            step.

        Returns:
          Next DFA state. If no outgoing edge guard is satisfied, returns the
          original ``state`` (i.e., an implicit self-loop).
        """
        for next_state in self.edges[state].keys():
            if self.edges[state][next_state].sat(labels):
                return next_state
        return state

    def step(self, labels: Iterable[str]) -> Tuple[bool, int]:
        """Advances the DFA by one step using the provided labels.

        This updates the internal :attr:`state`.

        Args:
          labels: Iterable of atomic proposition names holding at the current
            step.

        Returns:
          A pair ``(accepting, state)`` where ``accepting`` indicates whether
          the new state is in :attr:`accepting`, and ``state`` is the updated
          automaton state.
        """
        next_state = self.transition(self.state, labels)
        self.state = next_state
        return self.state in self.accepting, self.state

    @property
    def num_automaton_states(self):
        """Returns the number of states in the automaton.

        Returns:
          ``len(self.states)``.
        """
        return len(self.states)

    @property
    def automaton_state(self):
        """Returns the number of states in the automaton.

        Returns:
          ``len(self.states)``.
        """
        return self.state


def dfa_to_costfn(dfa: DFA):
    """Wraps a DFA as a :class:`DFACostFn` via a deep copy.

    Args:
      dfa: The DFA to wrap.

    Returns:
      A :class:`DFACostFn` wrapper around a deep-copied DFA.

    Notes:
      The deep copy prevents unexpected side effects if the caller later mutates
      the original DFA (e.g., by adding edges).
    """
    return DFACostFn(deepcopy(dfa))


class DFACostFn(DFA, CostFn):
    r"""DFA-backed MASA cost function.

    This wrapper interprets accepting automaton states as constraint violations
    (or terminal "bad" states): a transition that lands in an accepting state
    yields cost ``1.0`` and otherwise ``0.0``.

    Important:
      - The internal DFA state is advanced by calling :meth:`__call__`.
      - Use :meth:`cost` for counterfactual evaluation from an explicit DFA
        state without mutating internal state.
      - :meth:`DFA.step` is intentionally disabled to avoid ambiguous state
        updates via the inherited :class:`DFA` interface.

    Attributes:
      dfa: The wrapped DFA instance whose internal state is advanced when the
        cost function is called.
    """

    def __init__(self, dfa: DFA):
        """Creates a DFA cost function wrapper.

        Args:
          dfa: The DFA to wrap. The wrapper keeps a reference to this DFA and
            uses its internal state for sequential evaluation.
        """
        self.dfa = dfa
        self.states = self.dfa.states
        self.initial = self.dfa.initial
        self.accepting = self.dfa.accepting
        self.edges = self.dfa.edges

    def add_edge(self, parent: int, child: int, condition: Formula):
        """Disables edge modification after wrapping.

        Raises:
          RuntimeError: Always raised. Build the DFA fully before wrapping it as
            a cost function to avoid unintended side effects.
        """
        raise RuntimeError(
            "Please build the DFA before wrapping it as a cost function to avoid unintended side effects"
        )

    def reset(self):
        """Resets the internal DFA state.

        Notes:
          This delegates to the wrapped DFA's :meth:`DFA.reset`.
        """
        self.dfa.reset()

    def step(self, labels: Iterable[str]) -> Tuple[bool, int]:
        """Disables stepping via the DFA interface.

        Raises:
          RuntimeError: Always raised. Use :meth:`__call__` to advance the
            internal DFA and return the cost signal.
        """
        raise RuntimeError(
            "Please do not modify the the internal dfa state here, use DFACostFn.__call__ instead for correct functionality"
        )

    def cost(self, state: int, labels: Iterable[str]) -> float:
        """Computes the one-step cost from an explicit DFA state without mutation.

        Args:
          state: The DFA state to evaluate from (does not need to equal the
            internal automaton state).
          labels: Iterable of atomic proposition names for the current step.

        Returns:
          ``1.0`` iff the next state reached from ``state`` under ``labels`` is
          accepting; otherwise ``0.0``.

        Notes:
          This is intended for counterfactual evaluation and does not change the
          wrapped DFA's internal state.
        """
        return float(self.dfa.transition(state, labels) in self.dfa.accepting)

    def __call__(self, labels: Iterable[str]):
        """Advances the internal DFA by one step and returns the cost.

        Args:
          labels: Iterable of atomic proposition names for the current step.

        Returns:
          ``1.0`` if the internal DFA transitions into an accepting state on
          this step, else ``0.0``.
        """
        accepting, _ = self.dfa.step(labels)
        return float(accepting)

    @property
    def automaton_state(self):
        """Returns the current state of the wrapped DFA.

        Returns:
          The wrapped DFA's current automaton state (:attr:`DFA.state`).
        """
        return self.dfa.automaton_state


class ShapedCostFn(DFACostFn):
    r"""Potential-based shaped DFA cost for counterfactual experience.

    This class implements a potential-based shaping term on top of the base DFA
    cost, intended for counterfactual computations where you explicitly pass a
    DFA state to :meth:`DFACostFn.cost`.

    The shaped cost is:

    .. math::

       c'(q, \ell) = c(q, \ell) + \gamma \Phi(q') - \Phi(q),

    where :math:`q'` is the next automaton state after reading labels
    :math:`\ell`, :math:`c(q, \ell)` is the base DFA cost, and :math:`\Phi` is a
    user-provided potential function.

    Important:
      - This cost function is not intended to be used statefully.
      - :meth:`reset` and :meth:`__call__` are disabled by design.
    """

    def __init__(self, dfa: DFA, potential_fn: Callable[[int], float], gamma: float = 0.99):
        """Creates a shaped DFA cost function.

        Args:
          dfa: DFA whose accepting states define the base cost.
          potential_fn: Potential function :math:`\Phi(q)` over DFA states.
          gamma: Discount factor :math:`\gamma` used in potential-based shaping.
        """
        super().__init__(dfa)
        self.potential_fn = potential_fn
        self._gamma = gamma

    def reset(self):
        """Disables resetting for shaped counterfactual cost.

        Raises:
          RuntimeError: Always raised. This object is intended for counterfactual
            calls to :meth:`DFACostFn.cost` only.
        """
        raise RuntimeError(
            "Shaped cost function is not supposed to be reset only used for counter factual experiences"
        )

    def cost(self, state: int, labels: Iterable[str]) -> float:
        r"""Computes shaped cost from an explicit DFA state without mutation.

        The shaped cost is:

        .. math::

           c(q,\ell) + \gamma \Phi(q') - \Phi(q),

        where :math:`q' = \delta(q, \ell)` is the DFA transition result.

        Args:
          state: DFA state :math:`q` to evaluate from.
          labels: Iterable of atomic proposition names :math:`\ell` for the
            current step.

        Returns:
          Potential-based shaped cost.

        Notes:
          This method does not change the wrapped DFA's internal state.

        Warning:
          The implementation calls ``self.potential(state)`` for the final term,
          which assumes a method/attribute named ``potential`` exists. If you
          intended to use the provided callable, replace that with
          ``self.potential_fn(state)``.
        """
        next_state = self.dfa.transition(state, labels)
        cost = float(next_state in self.dfa.accepting)
        return cost + self._gamma * self.potential_fn(next_state) - self.potential(state)

    def __call__(self):
        """Disables stateful calling for shaped cost.

        Raises:
          RuntimeError: Always raised. This object is intended for counterfactual
            calls to :meth:`DFACostFn.cost` only.
        """
        raise RuntimeError(
            "Shaped cost function is not supposed to be called only used for counter factual experiences"
        )
