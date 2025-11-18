from __future__ import annotations
from typing import Iterable, List, Tuple
from masa.common.constraints.base import CostFn

class Formula:
    """Base class for propsoitional formula"""

    def sat(self, labels: Iterable[str]) -> bool:
        raise NotImplementedError("Propositional formula must implement a satisfaction relation")

class Atom(Formula): 
    """Atom: satisfied when the given atom is in the set of labels"""

    def __init__(self, atom: str):
        self.atom = atom

    def sat(self, labels: Iterable[str]) -> bool:
        return self.atom in labels

class Truth(Formula): 
    """Truth: always satisfied"""

    def __init__(self):
        pass

    def sat(self, labels: Iterable[str]) -> bool:
        return True

class And(Formula):
    """And: satisfied when both subformulae are satisfied"""

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        return self.subformula_1.sat(labels) and self.subformula_2.sat(labels)

class Or(Formula):
    """Or: satisfied when either subformulae are satisfied"""

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        return self.subformula_1.sat(labels) or self.subformula_2.sat(labels)

class Neg(Formula):
    """Negation: satisfied when the subformula is not satisfied"""
    
    def __init__(self, subformula: Formula):
        self.subformula = subformula
        
    def sat(self, labels: Iterable[str]) -> bool:
        return not self.subformula.sat(labels)

class Implies:
    """Implies: satisfied when subformula_2 is satisified if subformula_1 is satisfied"""

    def __init__(self, subformula_1: Formula, subformula_2: Formula):
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    def sat(self, labels: Iterable[str]) -> bool:
        return Or(Neg(self.subformula_1), self.subformula_2).sat(labels)

class DFA:

    """Implements a deterministic finite automata (DFA), 
       where state transitions are governed by propositional formula

    Input attributes:
        states: list of automata states
        initial: the initial state
        accepting: list of accepting states

    Other attributes:
        edges: dictionary of state to state transitions for each state
        state: current state of the DFA during execution
    """

    def __init__(self, states: List[int], initial: int, accepting: List[int]):

        self.states = states
        self.initial = initial
        self.accepting = accepting
        self.edges = {s : {} for s in self.states}
        self.reset()

    def add_edge(self, parent: int, child: int, condition: Formula):
        """adds an edge from parent to child"""
        self.edges[parent][child] = condition

    def reset(self) -> int:
        """resets the DFA to the initial state"""
        self.state = self.initial
        return self.state

    def has_edge(self, state_1: int, state_2: int) -> bool:
        """check if there is an edge from state_1 to state_2"""
        try: 
            x = self.edges[state_1][state_2]
            return True
        except KeyError:
            return False

    def check(self, trace: Iterable[Iterable[str]]) -> bool:
        """check if a given trace is accepted on the DFA"""
        state = self.initial
        for labels in trace:
            state = self.transition(state, labels)
        return state in self.accepting

    def transition(self, state: int, labels: Iterable[str]) -> int:
        """compute the next state from a given state and set of labels"""
        for next_state in self.edges[state].keys():
            if self.edges[state][next_state].sat(labels):
                return next_state
        return state

    def step(self, labels: Iterable[str]) -> Tuple[bool, int]:
        """evolve the DFA one step for a given set of labels"""
        next_state = self.transition(self.state, labels)
        self.state = next_state
        return self.state in self.accepting, self.state

    @property
    def num_automaton_states(self):
        """returns the number of automaton states"""
        return len(self.states)

    @property
    def automaton_state(self):
        """returns the current dfa state"""
        return self.state

def dfa_to_costfn(dfa: DFA):
    return DFACostFn(dfa)

class DFACostFn(DFA, CostFn):

    """Implements a DFA cost function, where cost=1.0 for accepting automaton states"""

    def __init__(self, dfa: DFA):
        self.dfa = dfa

    def add_edge(self, parent: int, child: int, condition: Formula):
        raise RuntimeError("Please build the DFA before wrapping it as a cost function to avoid unintended side effects")

    def reset(self):
        self.dfa.reset()

    def has_edge(self, state_1: int, state_2: int) -> bool:
        return self.dfa.has_edge(state_1, state_2)

    def check(self, trace: Iterable[Iterable[str]]) -> bool:
        return self.dfa.check(trace)

    def transition(self, state: int, labels: Iterable[str]) -> int:
        """compute the next state from a given state and set of labels"""
        return self.dfa.transition(state, labels)

    def step(self, labels: Iterable[str]) -> Tuple[bool, int]:
        raise RuntimeError("Please do not modify the the internal dfa state here, use DFACostFn.__call__ instead for correct functionality")

    def cost(self, state: int, labels: Iterable[str]) -> float:
        """compute the cost from a given state and set of labels"""
        return float(self.dfa.transition(state, labels) in self.dfa.accepting)

    def __call__(self, labels: Iterable[str]):
        """steps the inetrnal dfa and returns cost=1.0 if accepting"""
        accepting, _ = self.dfa.step(labels)
        return float(accepting)

    @property
    def num_automaton_states(self):
        """returns the number of automaton states"""
        return self.dfa.num_automaton_states

    @property
    def automaton_state(self):
        """returns the current internal dfa state"""
        return self.dfa.automaton_state

class ShapedCostFn(DFACostFn):

    def __init__(self, dfa: DFA, potential_fn: Callable[int, float], gamma: float = 0.99):
        super().__init__(dfa)
        self.potential_fn = potential_fn
        self._gamma = gamma

    def reset(self):
        raise RuntimeError("Shaped cost function is not supposed to be reset only used for counter factual experiences")

    def cost(self, state: int, labels: Iterable[str]) -> float:
        next_state = self.dfa.transition(state, labels)
        cost = float(next_state in self.dfa.accepting)
        return cost + self._gamma * self.potential_fn(next_state) - self.potential(state)

    def __call__(self):
        raise RuntimeError("Shaped cost function is not supposed to be called only used for counter factual experiences")
    