Linear Temporal Logic (LTL)
==========================

Overview: Safety LTL in MASA (DFA + Costs)
-----------------------------------------

MASA focuses on **safety specifications** expressed in the *safety fragment* of
Linear Temporal Logic (LTL). A safety specification is monitored online by
converting it to a **deterministic finite automaton (DFA)** and executing that
automaton over the environment’s **label trace** (sets of atomic propositions).

The same DFA can also be wrapped as a **cost function**, producing a scalar
safety signal suitable for reinforcement learning.

In practice, MASA uses:

- **A labelling function**  
  Environment state/observation → set of atomic proposition names (strings).

- **Propositional guard formulae** on DFA edges  
  For example ``Atom``, ``And``, ``Or``, ``Neg``, ``Implies``, and ``Truth``.

- **A DFA monitor**  
  Consumes the label trace and tracks progress or violation of the specification.

- **A DFA cost function**  
  ``DFACostFn`` returns ``1.0`` when a violation is detected (reaching an
  accepting *bad* state), and ``0.0`` otherwise.

The key design choice in MASA is that **accepting DFA states are interpreted as
violation states** for safety monitoring. This makes “a violation occurred”
directly convertible into a per-step cost signal.

Mathematical Details
--------------------

LTL Traces and Satisfaction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{Let } AP \text{ be a finite set of atomic propositions.}

A *label* is a subset :math:`L \subseteq AP`. A (possibly infinite) *trace* is a
sequence of labels:

.. math::

   \rho = L_0 L_1 L_2 \ldots \in (2^{AP})^\omega.

An LTL formula is interpreted over traces. MASA’s implementation uses
**propositional formulae** as building blocks (edge guards), evaluated at a
single time step:

.. math::

   g : 2^{AP} \rightarrow \{\mathsf{true}, \mathsf{false}\}.

Concretely, a propositional guard ``g`` is satisfied by labels ``L`` iff
``g.sat(L)`` returns ``True``.

Safety Fragment and DFA Monitors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Informally, a property is a **safety property** if *something bad never happens*.
Equivalently, if the property is violated, there exists a **finite bad prefix**
that irrevocably witnesses the violation.

A standard result in automata-theoretic verification is that LTL formulas can be
compiled into automata over :math:`2^{AP}`. For **safety LTL**, one can construct
a **deterministic monitor automaton** that detects bad prefixes. Once such a
monitor reaches a designated bad state, no continuation of the trace can repair
the violation.

MASA assumes (either via downstream tooling or hand-built examples) a DFA:

.. math::

   \mathcal{A} = (Q, q_0, F, \delta),

where:

- :math:`Q` is a finite set of automaton states,
- :math:`q_0 \in Q` is the initial state,
- :math:`F \subseteq Q` is a set of **accepting states**, interpreted in MASA as
  *violation states* for safety,
- :math:`\delta` is a transition function driven by labels.

DFA Transitions Guarded by Propositional Formulae
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of defining :math:`\delta` as a raw transition table over
:math:`2^{AP}`, MASA represents outgoing transitions from a state :math:`q` as
**guarded edges**:

- Each edge :math:`(q \rightarrow q')` is associated with a propositional guard
  formula :math:`g_{q,q'}`.
- At runtime, the transition taken is the first outgoing edge whose guard is
  satisfied by the current label set.

Formally:

.. math::

   \delta(q, L) =
   \begin{cases}
     q' & \text{for the first } q' \text{ such that }
           g_{q,q'}(L) = \mathsf{true}, \\
     q  & \text{if no guard is satisfied (implicit self-loop).}
   \end{cases}

This representation makes DFA construction readable and modular: guards are
built from propositional connectives, and transitions are added explicitly using
``Formula`` objects.

MASA LTL Pipeline (High-Level)
------------------------------

1. **Environment state → labels (atomic propositions)**  
   Environments expose a *labelling function* that maps an observation or state
   to a set of atomic proposition names (strings), for example
   ``{"unsafe"}``, ``{"goal"}``, or ``{"bomb"}``.

2. **Propositional formulae guard transitions**  
   DFA edges are guarded by lightweight propositional ``Formula`` objects such
   as ``Atom``, ``And``, ``Or``, ``Neg``, ``Implies``, and ``Truth``.

3. **DFA execution**  
   The DFA consumes the label trace step-by-step, updating its internal state to
   reflect progress or violation of the safety specification.

4. **DFA → cost function**  
   MASA wraps DFAs as constraint costs using ``DFACostFn``. In this convention:

   - The cost is ``1.0`` if the DFA transitions into an accepting (violation)
     state at the current step.
   - The cost is ``0.0`` otherwise.

   Two evaluation modes are supported:

   - **Stateful monitoring** via ``DFACostFn(labels)``, which steps the internal
     DFA.
   - **Counterfactual / offline evaluation** via
     ``DFACostFn.cost(state, labels)``, which computes the one-step cost from an
     explicit automaton state *without* mutating internal state.

5. **Shaped costs for counterfactual experience**  
   For certain algorithms (e.g., counterfactual rollouts or shaping for
   exploration), MASA provides ``ShapedCostFn``, which augments the base DFA cost
   with a potential-based shaping term:

   .. math::

      c'(q, L) = c(q, L) + \gamma \Phi(\delta(q, L)) - \Phi(q).

   This shaped cost is intentionally **not stateful** and is meant to be queried
   using explicit automaton states during counterfactual computations.

Next Steps
----------

- `**Propositional Formula** <LTL/Propositional%20Formula>_`
- `**Deterministic Finite Automata (DFA)** <LTL/DFA>_`
- `**Cost Function as DFA** <LTL/Cost%20Function%20as%20DFA>_`
- `**Shaped Cost Function** <LTL/Shaped%20Cost%20Function>_`


.. toctree::
   :caption: Linear Temporal Logic (LTL)
   :hidden:

   LTL/Propositional Formula
   LTL/DFA
   LTL/Cost Function as DFA
   LTL/Shaped Cost Function
