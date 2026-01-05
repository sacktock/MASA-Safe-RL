Probabilistic Shielding
=========================

This module provides a **Gymnasium-compatible implementation of Probabilistic Shielding**
for Safe Reinforcement Learning, based on the state-augmentation framework introduced in:

**Probabilistic Shielding for Safe Reinforcement Learning**  
Edwin Hamel-De le Court, Francesco Belardinelli, Alexander W. Goodall  
arXiv: https://arxiv.org/abs/2503.07671

The approach guarantees **probabilistic safety during both training and evaluation**, while
remaining **optimality-preserving** among all safe policies.

Overview
--------

Probabilistic Shielding addresses reinforcement learning problems of the form:

*Maximise discounted reward subject to an undiscounted probabilistic safety constraint.*

Safety is expressed as an **avoidance property**:

.. math::

    \mathbb{P}(\text{reach unsafe}) \le p

Rather than constraining the policy directly, the method constructs a
**safety-aware augmented MDP** (the *shield*) in which **every policy is provably safe**.

Any standard RL algorithm (e.g. PPO) can then be trained on the shielded environment.

Generic Procedure
-----------------

Given an environment with known **safety dynamics**:

1. **Compute safety bounds**  
   Use sound value iteration (interval iteration) to compute an *inductive upper bound*
   :math:`\beta(s)` on the minimal probability of reaching an unsafe state from each state.

2. **Augment the MDP**  
   Each state is augmented with a *remaining safety budget*
   :math:`q \in [\beta(s), 1]`.

3. **Restrict actions via a shield**  
   At each augmented state :math:`(s, q)`, only actions that **provably preserve the safety
   bound** are allowed. This is enforced by projecting agent-selected actions onto a safe
   probability simplex.

4. **Train normally**  
   Train any RL algorithm on the shielded MDP.  
   Safety is guaranteed **by construction**, not by penalties or Lagrangians.


The ``ProbShieldWrapperDisc``
----------------------------

The main entry point is the Gymnasium wrapper:

.. code-block:: python

    ProbShieldWrapperDisc(env, ...)

Expected inputs and types:

.. code-block:: python

    env: TabularEnv | DiscreteEnv
        Must expose safety dynamics either as a full transition kernel or compact kernel:
          successor_states_matrix: np.ndarray[int]  # (K, n_states)
          probabilities: np.ndarray[float]          # (K, n_states, n_actions)

    label_fn: Callable[[int], Labels]
        Called on discrete state ids (or abstract ids if using safety_abstraction).

    cost_fn: Callable[[Labels], float]
        MASA CostFn. Maps labels (set of atomic predicates) -> float cost.

    safety_abstraction: Optional[Callable[[Any], int]]
        Maps raw env observations/states -> discrete abstract state id.
        Required if observation_space is not Discrete.

The wrapper:

- Computes safety bounds automatically using interval value iteration
- Augments observations with the current safety budget
- Projects actions to satisfy the probabilistic safety constraint
- Is compatible with **any on-policy or off-policy RL algorithm**

Implementation details can be found in ``prob_shield_wrapper_disc.py``.

Usage Examples
--------------

Basic Probabilistic Shielding (Discrete MDP, PCTL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For environments with discrete state spaces and PCTL safety constraints:

.. code-block:: python

    env = make_env(
        "pacman",
        "pctl",
        1000,
        label_fn=label_fn,
        cost_fn=cost_fn,
        alpha=0.01,
    )

    env = ProbShieldWrapperDisc(
        env,
        init_safety_bound=0.01,
        theta=1e-15,
        max_vi_steps=10_000,
        granularity=20,
    )

See the full example in ``prob_shield_example.py``.

Probabilistic Shielding with a Safety Abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large or combinatorial environments, you can provide a **discrete safety abstraction**
that preserves only safety-relevant dynamics.

.. code-block:: python

    env = ProbShieldWrapperDisc(
        env,
        label_fn=abstr_label_fn,
        cost_fn=cost_fn,
        safety_abstraction=safety_abstraction,
        init_safety_bound=0.01,
    )

This enables scalable safety verification even when the full state space is very large.

See ``prob_shield_safety_abstraction_example.py``.

Probabilistic Shielding for Safety-LTL (DFA–MDP Product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Safety properties expressed in **Safety-LTL** are handled by constructing the
**DFA–MDP product** internally.

.. code-block:: python

    env = make_env(
        "colour_bomb_grid_world_v2",
        "ltl_dfa",
        250,
        label_fn=label_fn,
        dfa=make_dfa(),
    )

    env = ProbShieldWrapperDisc(env, init_safety_bound=0.01)

The shield is built over the **DFA–MDP product**, ensuring probabilistic satisfaction
of the LTL safety property.

See ``prob_shield_ltl_example.py``.

When to Use
-----------

Use Probabilistic Shielding when:

- Safety is **non-negotiable**
- Constraints are **probabilistic**, not expected-cost based
- You want **formal guarantees**, not penalties or Lagrangians
- The safety dynamics (or a conservative abstraction) are known

Citation
--------

If you use this implementation, please cite:

.. code-block:: bibtex

    @article{hamel2025probabilistic,
      title={Probabilistic Shielding for Safe Reinforcement Learning},
      author={Hamel-De le Court, Edwin and Belardinelli, Francesco and Goodall, Alexander W.},
      journal={arXiv preprint arXiv:2503.07671},
      year={2025}
    }