
def dfa_to_costfn(dfa)
    return DFACostFn(dfa)

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

    def __init__(self, states, initial, accepting):

        assert type(states) is list
        assert initial in states
        assert type(accepting) is list
        self.states = states
        self.initial = initial
        self.accepting = accepting
        self.edges = {s : {} for s in self.states}
        self.reset()

    def add_edge(self, parent, child, condition):
        """adds an edge from parent to child"""
        self.edges[parent][child] = condition

    def reset(self):
        """resets the DFA to the initial state"""
        self.state = self.initial
        return self.state

    def has_edge(self, state_1, state_2):
        """check if there is an edge from state_1 to state_2"""
        try: 
            x = self.edges[state_1][state_2]
            return True
        except KeyError:
            return False

    def check(self, trace):
        """check if a given trace is accepted on the DFA"""
        state = self.initial
        for labels in trace:
            state = self.transition(state, labels)
        return state in self.accepting

    def transition(self, state, labels):
        """compute the next state from a given state and set of labels"""
        for next_state in self.edges[state].keys():
            if self.edges[state][next_state].sat(labels):
                return next_state
        return state

    def step(self, labels):
        """evolve the DFA one step for a given set of labels"""
        next_state = self.transition(self.state, labels)
        self.state = next_state
        return self.state in self.accepting, self.state

class DFACostFn(DFA):

    """Implements a DFA cost function, where cost=1.0 is supplied for accepting automaton states"""

    def __init__(self, dfa):
        self.dfa = dfa

    def reset(self):
        self.dfa.reset()

    def __call__(self, info):
        labels = info["labels"]
        accepting, _ = self.dfa.step(labels)
        return float(accepting)

    @property
    def automaton_state(self):
        return self.dfa.state