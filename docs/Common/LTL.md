# Linear Temporal Logic (LTL)

MASA supports **safety specifications** written in (Safety-)LTL by compiling them into a
**deterministic finite automaton (DFA)**, and then using the DFA either:

1. **As a monitor** that tracks satisfaction/violation along an environment trajectory, or
2. **As a cost function** (via `DFACostFn`) that produces a scalar safety signal during RL.

At a high level, MASA's LTL pipeline looks like this:

1. **Environment state -> labels (atomic propositions)**  
   Environments expose a *labelling function* that maps an observation/state to a set of
   atomic proposition names (strings), e.g. `{"unsafe"}`, `{"goal"}`, `{"bomb"}`, etc.
   These labels are the interface between the environment and the logic machinery.

2. **Propositional formulae guard transitions**  
   MASA represents edge guards in automata using lightweight propositional formula objects:
   `Atom`, `And`, `Or`, `Neg`, `Implies`, and `Truth`.  
   Each formula defines a satisfaction predicate:
   `sat(labels) -> bool`, meaning "does this set of propositions satisfy the guard?"

3. **Safety LTL -> DFA**  
   Safety LTL specifications are handled by converting them into a DFA (either constructed
   manually for examples, or produced by a compiler in downstream tooling).  
   The DFA consumes a **trace of labels**: at each time step, the current label set is used
   to choose an enabled outgoing transition (whose guard formula is satisfied). The DFA's
   internal state acts as a compact summary of property progress.

4. **DFA -> cost function**  
   MASA wraps DFAs as constraint costs using `DFACostFn`. In this convention, reaching an
   **accepting DFA state** indicates a *violation / terminal bad condition* (common for
   safety monitoring), and the cost is:
   - `1.0` if the DFA transitions into an accepting state at the current step
   - `0.0` otherwise

   Two evaluation modes are supported:
   - **Stateful monitoring** via `DFACostFn(labels)` which steps the internal DFA.
   - **Counterfactual / offline evaluation** via `DFACostFn.cost(state, labels)` which
     computes the one-step cost from an explicit DFA state *without* mutating internal state.

5. **Shaped costs for counterfactual experience**  
   For certain algorithms (e.g., counterfactual rollouts, shaping for exploration), MASA
   provides `ShapedCostFn`, which adds a potential-based shaping term on top of the base DFA
   cost:
   `base_cost + gamma * Phi(next_state) - Phi(state)`.

   This shaped cost is intentionally **not** stateful: it is meant to be queried with
   explicit automaton states during counterfactual computations.

#### Next Steps

-[Propositional Formula](https://sacktock.github.io/MASA-Safe-RL/Common/LTL/Propositional%20Formula)
-[Deterministic Finite Automata (DFA)](https://sacktock.github.io/MASA-Safe-RL/Common/LTL/DFA)
-[Cost Function as DFA](https://sacktock.github.io/MASA-Safe-RL/Common/LTL/Cost%20Function%20as%20DFA)
-[Shaped Cost Function](https://sacktock.github.io/MASA-Safe-RL/Common/LTL/Shaped%20Cost%20Function)

```{toctree}
:caption: Linear Temporal Logic (LTL)
:hidden:

LTL/Propositional Formula
LTL/DFA
LTL/Cost Function as DFA
LTL/Shaped Cost Function
```

