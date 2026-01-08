Probabilistic Computation Tree Logic (PCTL)
===========================================

Overview
--------

MASA supports **bounded Probabilistic Computation Tree Logic (PCTL)** for reasoning
about *probabilistic safety and liveness properties* of Markov decision processes
(MDPs) and the Markov chains induced by fixing a policy.

The implementation focuses on:

- **Bounded temporal operators** for finite-horizon reasoning.
- **Vectorized evaluation** of atomic predicates via a labeling function.
- **JAX-accelerated computation** for scalable probability-sequence evaluation.
- **Dense and compact transition-kernel representations** to handle both
  small tabular and large sparse dynamics.

Bounded PCTL formulas in MASA
-----------------------------

MASA implements a *bounded* fragment of PCTL: all temporal operators are evaluated
over a finite horizon, which aligns naturally with finite rollouts and Monte Carlo
estimation.

A bounded PCTL formula is represented as a tree of ``BoundedPCTLFormula`` objects and
falls into two broad categories.

State (propositional) formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

State formulas are Boolean formulas evaluated **at a single state**, without reference
to transitions. MASA supports:

- ``Truth``
- Atomic predicates (``Atom``)
- Boolean connectives: :math:`\neg`, :math:`\land`, :math:`\lor`, :math:`\to`

State formulas are evaluated using a **vectorized labeling representation**:

- A ``LabelFn`` maps each state to a set of atomic predicates (strings).
- Internally, MASA builds a dense matrix:

.. math::

   \mathrm{vec\_label\_fn} \in \{0, 1\}^{|AP| \times |S|}

where ``vec_label_fn[i, s] = 1`` iff atomic predicate ``AP[i]`` holds in state ``s``.

State formulas have **zero temporal bound** (they do not extend the horizon).

Probabilistic temporal formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temporal formulas reason about the probability that events occur along paths within
a bounded number of steps. MASA supports the following bounded operators.

Next
^^^^

.. math::

   \mathbb{P}_{\ge p}[\,X\ \Phi\,]

The formula holds in state ``s`` if the probability that :math:`\Phi` holds in the
*next* state is at least ``p``.

Until (bounded)
^^^^^^^^^^^^^^^

.. math::

   \mathbb{P}_{\ge p}[\,\Phi_1\ U^{\le B}\ \Phi_2\,]

The formula holds in state ``s`` if the probability that :math:`\Phi_2` becomes true
within ``B`` steps while :math:`\Phi_1` holds at all preceding steps is at least ``p``.

Eventually (syntactic sugar)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \mathbb{P}_{\ge p}[\,F^{\le B}\ \Phi\,] \equiv
   \mathbb{P}_{\ge p}[\,\mathrm{True}\ U^{\le B}\ \Phi\,]

Always (duality)
^^^^^^^^^^^^^^^^

.. math::

   \mathbb{P}_{\ge p}[\,G^{\le B}\ \Phi\,] \equiv
   \neg\,\mathbb{P}_{\geq 1-p}[\,\mathrm{True}\ U^{\le B}\ \neg\Phi\,]

Probability sequences and bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each probabilistic operator contributes a **local bound** and computes a per-state
**probability sequence** up to that horizon:

.. math::

   P_0(s), P_1(s), \dots, P_B(s)

where :math:`P_k(s)` is the probability that the subformula's path condition is
satisfied within :math:`k` steps starting from state :math:`s`.

Satisfaction for probabilistic operators is obtained by thresholding the probability
at the operator's bound:

.. math::

   \text{sat}(s) = \mathbf{1}\{P_B(s) \ge p\}

Transition kernels and policy-induced chains
--------------------------------------------

PCTL evaluation in MASA is performed on a **Markov chain kernel**. For an MDP, MASA
first fixes a policy and collapses the MDP into a Markov chain by taking expectations
over actions.

MASA supports two representations of the resulting transition kernel.

Dense kernel
~~~~~~~~~~~~

A dense Markov-chain transition matrix of shape ``(n_states, n_states)``.
Entry ``(s', s)`` gives the probability of transitioning **from** ``s`` **to** ``s'``
(i.e., columns represent source states).

This form is used when the full transition structure is available and reasonably sized.

Compact kernel
~~~~~~~~~~~~~~

A sparse successor representation that stores at most ``K`` successors per state:

- ``succ`` with shape ``(K, n_states)`` storing successor state ids.
- ``p`` with shape ``(K, n_states)`` storing the corresponding transition probabilities.

This form is efficient when each state has relatively few successors, even if the
overall state space is large.

In both cases, probabilistic operators (e.g., ``Next`` and bounded ``Until``) evaluate
expectations of the form :math:`\mathbb{E}[V(s')]` by either:

- matrix multiplication for dense kernels, or
- successor-index gathering and weighted sums for compact kernels.

Model checking in MASA
----------------------

MASA provides both exact and sampling-based checking behind a shared interface:

- If a transition model is available, MASA can evaluate the probability sequences
  deterministically by dynamic programming over the policy-induced Markov chain.
- If the model is unknown or too large, MASA can estimate satisfaction probabilities
  by Monte Carlo sampling of bounded trajectories under the policy, and then compare
  the estimate to the formula's probability threshold.

Pure state formulas (``Truth``, ``Atom``, and boolean connectives) can be evaluated
directly without sampling, while probabilistic temporal operators use either the
exact recurrence or sampled trajectory satisfaction depending on the chosen approach.

Next Steps
----------

- **`Propositional Operators </PCTL/Propositional%20Operators>`_**
- **`Temporal Operators </PCTL/Temporal%20Operators>`_**
- **`Model Checkin </PCTL/Model%20Checking>`_**

.. toctree::
   :caption: Probabilistic Computation Tree Logic (PCTL)
   :hidden:

   PCTL/Propsotional Operators
   PCTL/Temporal Opetrators
   PCTl/Model Checking
