# Linear Temporal Logic (LTL)

## Overview: Safety LTL in MASA (DFA + Costs)

MASA focuses on **safety** specifications expressed in the *safety fragment* of LTL. A safety specification is monitored online by converting it to a **deterministic finite automaton (DFA)** and running that DFA over the environment's **label trace** (sets of atomic propositions). The same DFA can be wrapped as a **cost function** to provide a scalar safety signal suitable for reinforcement learning.

In practice, MASA uses:
- **A labelling function**: environment state/observation -> set of atomic propositions (strings).
- **Propositional guard formulae** on DFA edges (e.g. `Atom`, `And`, `Or`, `Neg`, `Implies`, `Truth`).
- **A DFA monitor**: consumes the label trace and tracks progress/violation.
- **A DFA cost function**: `DFACostFn` returns `1.0` when a violation is detected (accepting "bad" state), else `0.0`.

The key design choice is that MASA treats **accepting DFA states as violation states** for safety monitoring, which makes "violation happened" easy to convert into a per-step cost signal.

## Mathematical details

### LTL traces and satisfaction

```{eval-rst}
Let :math:`AP` be a finite set of atomic propositions. A *label* is a subset :math:`L \subseteq AP`. A (possibly infinite) *trace* is a sequence of labels:

.. math::

   \rho = L_0 L_1 L_2 \ldots \in (2^{AP})^\omega.

An LTL formula is interpreted over traces. MASA's implementation uses **propositional formulae** as building blocks (edge guards), evaluated at a single step:

.. math::

   g : 2^{AP} \to {\mathsf{true}, \mathsf{false}}.

```

Concretely, a propositional guard `g` is satisfied by labels `L` iff `g.sat(L)` returns `True`.

### Safety fragment

Informally, a property is a **safety property** if "something bad never happens". Equivalently, if the property is violated, there exists a *finite bad prefix* witnessing the violation.

```{eval-rst}
.. math::

   \varphi \text{ is safety } \iff \forall \rho \not\models \varphi,; \exists k; \forall \rho' \in (2^{AP})^\omega:;
   (L_0 \ldots L_k \preceq \rho') \Rightarrow \rho' \not\models \varphi.

```

That is: once a finite prefix is "bad", no continuation can repair it. This is the reason safety monitoring works well with automata: you can detect violation after a finite amount of observation.

### Key result: Safety LTL -> DFA monitor

```{eval-rst}
A standard result in automata-theoretic verification is that LTL formulas can be compiled into automata over :math:`2^AP`. For safety LTL, one can construct a **deterministic** monitor automaton that detects bad prefixes (often phrased as a DFA over finite words, or as a deterministic monitor that reaches a sink "bad" state once violation is inevitable).

MASA assumes (either via downstream tooling or hand-built examples) a DFA:

.. math::

   \mathcal{A} = (Q, q_0, F, \delta),

where:

* :math:`Q` is a finite set of automaton states,
* :math:`q_0 \in Q` is the initial state,
* :math:`F \subseteq Q` is a set of **accepting states** (interpreted in MASA as *violation states* for safety),
* :math:`\delta` is a transition function driven by labels.

```

### DFA transitions guarded by propositional formulae

```{eval-rst}
Instead of defining :math:`\delta` as a raw table over :math:`2^AP`, MASA represents outgoing transitions from a state :math:`q` as **guarded edges**:

* An edge :math:`(q \to q')` has a guard formula :math:`g_{q,q'}`.
* The transition chosen at runtime is the first edge whose guard is satisfied by the current label set.

.. math::

   \delta(q, L) =
   \begin{cases}
   q' & \text{for the first } q' \text{ such that } g_{q,q'}(L)=\mathsf{true}, \
   q  & \text{if no guard is satisfied (implicit self-loop).}
   \end{cases}

This makes DFA construction readable and modular: guards are built from propositional connectives, and edges are added with a ``Formula`` object.
```

## MASA LTL pipeline (high-level)

1. **Environment state -> labels (atomic propositions)**  
   Environments expose a *labelling function* that maps an observation/state to a set of
   atomic proposition names (strings), e.g. `{"unsafe"}`, `{"goal"}`, `{"bomb"}`, etc.

2. **Propositional formulae guard transitions**  
   MASA represents edge guards in automata via lightweight propositional `Formula` objects:
   `Atom`, `And`, `Or`, `Neg`, `Implies`, and `Truth`.  

4. **DFA -> cost function**  
   MASA wraps DFAs as constraint costs using `DFACostFn`. In this convention, reaching an
   **accepting DFA state** indicates a *violation / terminal bad condition* (common for
   safety monitoring), and the cost is:
   - `1.0` if the DFA transitions into an accepting state at the current step
   - `0.0` otherwise

   Two evaluation modes are supported:
   - **Stateful monitoring** via :meth:`DFACostFn(labels)` which steps the internal DFA.
   - **Counterfactual / offline evaluation** via :meth:`DFACostFn.cost(state, labels)` which
     computes the one-step cost from an explicit DFA state *without* mutating internal state.

5. **Shaped costs for counterfactual experience**  
   For certain algorithms (e.g., counterfactual rollouts, shaping for exploration), MASA
   provides `ShapedCostFn`, which adds a potential-based shaping term on top of the base DFA
   cost:

```{eval-rst}
.. math::

   c'(q, L) = c(q, L) + \gamma \Phi(\delta(q, L)) - \Phi(q).

```
   This shaped cost is intentionally **not** stateful: it is meant to be queried with
   explicit automaton states during counterfactual computations.

#### Next Steps

- [Propositional Formula](https://sacktock.github.io/MASA-Safe-RL/Common/LTL/Propositional%20Formula)
- [Deterministic Finite Automata (DFA)](https://sacktock.github.io/MASA-Safe-RL/Common/LTL/DFA)
- [Cost Function as DFA](https://sacktock.github.io/MASA-Safe-RL/Common/LTL/Cost%20Function%20as%20DFA)
- [Shaped Cost Function](https://sacktock.github.io/MASA-Safe-RL/Common/LTL/Shaped%20Cost%20Function)

```{toctree}
:caption: Linear Temporal Logic (LTL)
:hidden:

LTL/Propositional Formula
LTL/DFA
LTL/Cost Function as DFA
LTL/Shaped Cost Function
```

