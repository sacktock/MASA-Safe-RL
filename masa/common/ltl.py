"""
ltl.py

Lightweight propositional (LTL-style) formula objects and a deterministic finite
automaton (DFA) whose transitions are guarded by those formulae.

The core idea is:

- A *label* is an iterable of atomic proposition names (strings).
- A formula implements a satisfaction predicate ``sat(labels) -> bool``.
- A DFA transition from state ``q`` to ``q'`` is enabled when the edge's formula
  is satisfied by the current label set.
- ``DFACostFn`` wraps a DFA as a MASA ``CostFn`` where the cost is ``1.0`` on
  accepting states (after stepping) and ``0.0`` otherwise.

This module is intentionally minimal: it does not implement full LTL temporal
operators (X, U, F, G). Instead, it provides propositional guards for DFA edges,
which is sufficient for DFA-based monitoring and safety cost construction.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, List, Tuple

from masa.common.constraints.base import CostFn

class Formula:
    """
    Base class for propositional formulae over atomic proposition labels.

    Subclasses must implement :meth:`sat`, which defines the satisfaction
    relation between a formula and a label set.

    Notes
    -----
    The operators provided here are propositional (Boolean) connectives only. 
    Temporal structure is represented externally via a DFA that consumes a trace of label sets.
    """

    def sat(self, labels: Iterable[str]) -> bool:
        """
        Check whether the formula is satisfied by the given labels.

        Parameters
        ----------
        labels:
            Iterable of atomic proposition names that hold at the current step.

        Returns
        -------
        bool
            True if the formula is satisfied under the given labels.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement the satisfaction relation.
        """
        raise NotImplementedError(
            "Propositional formula must implement a satisfaction relation"
        )

class Atom(Formula):
    """
    Atomic proposition.

    An ``Atom(a)`` is satisfied iff the string ``a`` is present in the provided
    label set.
    """

    def __init__(self, atom: str):
        """
        Create an atomic proposition formula.

        Parameters
        ----------
        atom:
            The name of the atomic proposition (e.g. ``"goal"`` or ``"unsafe"``).
        """
        self.atom = atom

    def sat(self, labels: Iterable[str]) -> bool:
        """
        Evaluate whether the atom is present in ``labels``.

        Parameters
        ----------
        labels:
            Iterable of atomic proposition names.

        Returns
        -------
        bool
            True iff ``self.atom`` is in ``labels``.
        """
        return self.atom in labels


class Truth(Formula):
    """
    Constant truth.

    Always satisfied, regardless of the labels.
    """

    def __init__(self):
        """Construct the constant-true formula."""
        pass

    def sat(self, labels: Iterable[str]) -> bool:
        """
        Evaluate the truth constant.

        Parameters
        ----------
        labels:
            Unused. Included for API consistency.

        Returns
        -------
        bool
            Always True.
        """
        return True


class And(Formula):
    """
    Conjunction of two subformulae.

    Satisfied iff both subformulae are satisfied.
    """

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        """
        Construct a conjunction.

        Parameters
        ----------
        subformula_1:
            Left operand.
        subformula_2:
            Right operand.
        """
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        """
        Evaluate conjunction satisfaction.

        Parameters
        ----------
        labels:
            Iterable of atomic proposition names.

        Returns
        -------
        bool
            True iff both subformulae are satisfied by ``labels``.
        """
        return self.subformula_1.sat(labels) and self.subformula_2.sat(labels)

class Or(Formula):
    """
    Disjunction of two subformulae.

    Satisfied iff at least one subformula is satisfied.
    """

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        """
        Construct a disjunction.

        Parameters
        ----------
        subformula_1:
            Left operand.
        subformula_2:
            Right operand.
        """
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        """
        Evaluate disjunction satisfaction.

        Parameters
        ----------
        labels:
            Iterable of atomic proposition names.

        Returns
        -------
        bool
            True iff either subformula is satisfied by ``labels``.
        """
        return self.subformula_1.sat(labels) or self.subformula_2.sat(labels)

class Neg(Formula):
    """
    Negation of a subformula.

    Satisfied iff the subformula is *not* satisfied.
    """
    
    def __init__(self, subformula: Formula):
        """
        Construct a negation.

        Parameters
        ----------
        subformula:
            The formula to negate.
        """
        self.subformula = subformula
        
    def sat(self, labels: Iterable[str]) -> bool:
        """
        Evaluate negation satisfaction.

        Parameters
        ----------
        labels:
            Iterable of atomic proposition names.

        Returns
        -------
        bool
            True iff ``subformula`` is not satisfied by ``labels``.
        """
        return not self.subformula.sat(labels)

class Implies(Formula):
    """
    Implication between two subformulae.

    ``Implies(a, b)`` is satisfied iff either ``a`` is false or ``b`` is true
    under the given labels, i.e. it is equivalent to ``¬a ∨ b``.
    """

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        """
        Construct an implication.

        Parameters
        ----------
        subformula_1:
            Antecedent (premise).
        subformula_2:
            Consequent (conclusion).
        """
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        """
        Evaluate implication satisfaction.

        Parameters
        ----------
        labels:
            Iterable of atomic proposition names.

        Returns
        -------
        bool
            True iff the implication holds under ``labels``.
        """
        return Or(Neg(self.subformula_1), self.subformula_2).sat(labels)

class DFA:

    """
    Deterministic finite automaton with propositional guards on edges.

    Each edge from a parent state to a child state is labelled with a formula
    (a propositional guard). A transition is taken when its guard is satisfied
    by the current label set.

    Attributes
    ----------
    states:
        List of automaton states.
    initial:
        The initial automaton state.
    accepting:
        List of accepting (final) states.
    edges:
        Transition structure: ``edges[parent][child] = condition``.
    state:
        Current automaton state used by :meth:`step`.

    Notes
    -----
    The transition function is *deterministic by convention*: if multiple
    outgoing guards from a state are simultaneously satisfied, the first one
    encountered in the iteration order will be taken. Ensure guards are
    mutually exclusive if determinism is required.
    """

    def __init__(self, states: List[int], initial: int, accepting: List[int]):
        """
        Create a DFA.

        Parameters
        ----------
        states:
            List of automaton states (typically integers).
        initial:
            Initial state.
        accepting:
            Accepting (final) states.
        """
        self.states = states
        self.initial = initial
        self.accepting = accepting
        self.edges = {s : {} for s in self.states}
        self.state = self.initial

    def add_edge(self, parent: int, child: int, condition: Formula):
        """
        Add a guarded transition ``parent -> child``.

        Parameters
        ----------
        parent:
            Source state.
        child:
            Destination state.
        condition:
            Guard formula enabling this transition when satisfied.

        Notes
        -----
        This overwrites any existing edge guard between the same parent/child
        pair.
        """
        self.edges[parent][child] = condition

    def reset(self) -> int:
        """
        Reset the DFA to its initial state.

        Returns
        -------
        int
            The reset state (i.e., ``self.initial``).
        """
        self.state = self.initial
        return self.state

    def has_edge(self, state_1: int, state_2: int) -> bool:
        """
        Check whether there is an edge ``state_1 -> state_2``.

        Parameters
        ----------
        state_1:
            Source state.
        state_2:
            Destination state.

        Returns
        -------
        bool
            True iff an explicit edge from ``state_1`` to ``state_2`` exists.
        """
        try: 
            x = self.edges[state_1][state_2]
            return True
        except KeyError:
            return False

    def check(self, trace: Iterable[Iterable[str]]) -> bool:
        """
        Check whether a trace is accepted by the DFA.

        Parameters
        ----------
        trace:
            Sequence of label sets, one per time step.

        Returns
        -------
        bool
            True iff the state reached after consuming the full trace is
            accepting.
        """
        state = self.initial
        for labels in trace:
            state = self.transition(state, labels)
        return state in self.accepting

    def transition(self, state: int, labels: Iterable[str]) -> int:
        """
        Compute the next automaton state given the current state and labels.

        Parameters
        ----------
        state:
            Current DFA state.
        labels:
            Iterable of atomic proposition names holding at the current step.

        Returns
        -------
        int
            Next DFA state. If no outgoing edge guard is satisfied, this
            returns the original ``state`` (i.e., a self-loop by default).
        """
        for next_state in self.edges[state].keys():
            if self.edges[state][next_state].sat(labels):
                return next_state
        return state

    def step(self, labels: Iterable[str]) -> Tuple[bool, int]:
        """
        Advance the DFA by one step using the provided labels.

        This updates the internal state ``self.state``.

        Parameters
        ----------
        labels:
            Iterable of atomic proposition names holding at the current step.

        Returns
        -------
        (bool, int)
            A pair ``(accepting, state)`` where ``accepting`` indicates whether
            the new state is accepting and ``state`` is the updated DFA state.
        """
        next_state = self.transition(self.state, labels)
        self.state = next_state
        return self.state in self.accepting, self.state

    @property
    def num_automaton_states(self):
        """
        Number of states in the automaton.

        Returns
        -------
        int
            ``len(self.states)``.
        """
        return len(self.states)

    @property
    def automaton_state(self):
        """
        Current internal automaton state.

        Returns
        -------
        int
            The DFA's current state used by :meth:`step`.
        """
        return self.state

def dfa_to_costfn(dfa: DFA):
    """
    Wrap a DFA as a :class:`DFACostFn` via a deep copy.

    Parameters
    ----------
    dfa:
        The DFA to wrap.

    Returns
    -------
    DFACostFn
        A cost-function wrapper around a deep-copied DFA.

    Notes
    -----
    The deep copy prevents unexpected side effects if the caller later mutates
    the original DFA (e.g., by adding edges).
    """
    return DFACostFn(deepcopy(dfa))

class DFACostFn(DFA, CostFn):
    """
    DFA-backed MASA cost function.

    This wrapper interprets *accepting* automaton states as constraint
    violations (or terminal "bad" states): a transition that lands in an
    accepting state yields cost ``1.0`` and otherwise ``0.0``.

    Important
    ---------
    - The internal DFA state is advanced by calling :meth:`__call__`.
    - Use :meth:`cost` for *counterfactual* evaluation from an explicit DFA
      state without mutating internal state.

    The method :meth:`step` is intentionally disabled to avoid ambiguous state
    updates through the inherited :class:`DFA` interface.
    """

    def __init__(self, dfa: DFA):
        """
        Create a DFA cost function wrapper.

        Parameters
        ----------
        dfa:
            The DFA to wrap. The wrapper keeps a reference to this DFA and uses
            its internal state for sequential evaluation.
        """
        self.dfa = dfa
        self.states = self.dfa.states
        self.initial = self.dfa.initial
        self.accepting = self.dfa.accepting
        self.edges = self.dfa.edges

    def add_edge(self, parent: int, child: int, condition: Formula):
        """
        Disable edge modification after wrapping.

        Raises
        ------
        RuntimeError
            Always raised. Build the DFA fully before wrapping it as a cost
            function to avoid unintended side effects.
        """
        raise RuntimeError("Please build the DFA before wrapping it as a cost function to avoid unintended side effects")

    def reset(self):
        """
        Reset the internal DFA state.

        Notes
        -----
        This delegates to the wrapped DFA's :meth:`reset`.
        """
        self.dfa.reset()

    def step(self, labels: Iterable[str]) -> Tuple[bool, int]:
        """
        Disable stepping via the DFA interface.

        Raises
        ------
        RuntimeError
            Always raised. Use :meth:`__call__` to advance the internal DFA and
            return the cost signal.
        """
        raise RuntimeError("Please do not modify the the internal dfa state here, use DFACostFn.__call__ instead for correct functionality")

    def cost(self, state: int, labels: Iterable[str]) -> float:
        """
        Compute the one-step cost from an explicit DFA state without mutation.

        Parameters
        ----------
        state:
            The DFA state to evaluate from (does not need to equal the internal
            automaton state).
        labels:
            Iterable of atomic proposition names for the current step.

        Returns
        -------
        float
            ``1.0`` iff the next state reached from ``state`` under ``labels``
            is accepting; otherwise ``0.0``.

        Notes
        -----
        This is intended for counterfactual evaluation and does not change the
        wrapped DFA's internal state.
        """
        return float(self.dfa.transition(state, labels) in self.dfa.accepting)

    def __call__(self, labels: Iterable[str]):
        """
        Advance the internal DFA by one step and return the cost.

        Parameters
        ----------
        labels:
            Iterable of atomic proposition names for the current step.

        Returns
        -------
        float
            ``1.0`` if the internal DFA transitions into an accepting state on
            this step, else ``0.0``.
        """
        accepting, _ = self.dfa.step(labels)
        return float(accepting)

    @property
    def automaton_state(self):
        """
        Current state of the wrapped DFA.

        Returns
        -------
        int
            The wrapped DFA's current automaton state.
        """
        return self.dfa.automaton_state

class ShapedCostFn(DFACostFn):
    """
    Potential-based shaped DFA cost for counterfactual experience.

    This class implements a potential-based shaping term on top of the base DFA
    cost, intended for *counterfactual* computations where you explicitly pass a
    DFA state to :meth:`cost`.

    Important
    ---------
    - This cost function is **not** intended to be used statefully.
    - :meth:`reset` and :meth:`__call__` are disabled by design.
    """

    def __init__(self, dfa: DFA, potential_fn: Callable[int, float], gamma: float = 0.99):
        """
        Create a shaped DFA cost function.

        Parameters
        ----------
        dfa:
            DFA whose accepting states define the base cost.
        potential_fn:
            Potential function :math:`\\Phi(q)` over DFA states.
        gamma:
            Discount factor used in potential-based shaping.
        """
        super().__init__(dfa)
        self.potential_fn = potential_fn
        self._gamma = gamma

    def reset(self):
        """
        Disable resetting for shaped counterfactual cost.

        Raises
        ------
        RuntimeError
            Always raised. This object is intended for counterfactual calls to
            :meth:`cost` only.
        """
        raise RuntimeError("Shaped cost function is not supposed to be reset only used for counter factual experiences")

    def cost(self, state: int, labels: Iterable[str]) -> float:
        """
        Compute shaped cost from an explicit DFA state without mutation.

        The shaped cost is:

        ``base_cost(next_state) + gamma * Phi(next_state) - Phi(state)``

        where ``base_cost(next_state)`` is ``1.0`` iff ``next_state`` is
        accepting.

        Parameters
        ----------
        state:
            DFA state to evaluate from.
        labels:
            Iterable of atomic proposition names for the current step.

        Returns
        -------
        float
            Potential-based shaped cost.

        Notes
        -----
        This method does not change the wrapped DFA's internal state.

        Raises
        ------
        AttributeError
            If the implementation references a non-existent attribute/method
            (e.g., ``self.potential`` instead of ``self.potential_fn``). Ensure
            the shaping term uses ``self.potential_fn(state)`` consistently.
        """
        next_state = self.dfa.transition(state, labels)
        cost = float(next_state in self.dfa.accepting)
        return cost + self._gamma * self.potential_fn(next_state) - self.potential(state)

    def __call__(self):
        """
        Disable stateful calling for shaped cost.

        Raises
        ------
        RuntimeError
            Always raised. This object is intended for counterfactual calls to
            :meth:`cost` only.
        """
        raise RuntimeError("Shaped cost function is not supposed to be called only used for counter factual experiences")
    