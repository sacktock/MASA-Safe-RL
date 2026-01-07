Labelling Function
==================

This page describes the **labelling function API**, the underlying **labelled Markov Decision Process (MDP)**
formalism, and the MASA convention for mapping environment observations to **sets of atomic predicates**.
These components are foundational for safety objectives, probabilistic shielding, and temporal-logic
specifications (e.g. safety-LTL).

Atomic Predicates and Labels
----------------------------

We assume a fixed (finite) set of **atomic predicates**:

.. math::

   \mathcal{AP} = \{p_1, p_2, \dots, p_k\}.

An atomic predicate represents a Boolean property of the environment state, such as:

- ``"unsafe"``
- ``"goal"``
- ``"collision"``
- ``"near_obstacle"``

At runtime, a state may satisfy **zero or more** atomic predicates simultaneously; therefore the label of a
state/observation is a **set** of predicates.

Labelling Function API
----------------------

Type Signature
~~~~~~~~~~~~~~

The labelling function type is:

.. code-block:: python

   LabelFn = Callable[[Any], Iterable[str]]

Formally, MASA treats labelling as a map from observations to predicate sets:

.. math::

   L : \mathcal{O} \rightarrow 2^{\mathcal{AP}},

where :math:`\mathcal{O}` is the observation space and :math:`2^{\mathcal{AP}}` is the power set of
atomic predicates.

Semantics
~~~~~~~~~

Given an observation ``obs``, the labelling function returns the **set of atomic predicates that hold** in
that observation.

Requirements:

- The output must be **iterable** (e.g. ``list``, ``tuple``, ``set``).
- Elements must be **strings**, each naming an atomic predicate in :math:`\mathcal{AP}`.
- The returned collection is interpreted as a **set** (duplicates are ignored).

Example
~~~~~~~

.. code-block:: python

   def label_fn(obs):
       labels = set()

       if obs["x"] < 0:
           labels.add("unsafe")

       if obs["goal_reached"]:
           labels.add("goal")

       return labels

Observation-to-Labels Convention
--------------------------------

MASA uses the following convention:

.. important::

   **Labels are computed from observations, not from internal environment state.**

This improves:

- Compatibility with partially observable environments,
- Consistency under wrappers/abstractions,
- Compositionality with automata- and logic-based monitors.

.. list-table:: Convention summary
   :header-rows: 1
   :widths: 28 72

   * - Concept
     - Convention
   * - Input to label function
     - Raw observation returned by :meth:`gymnasium.Env.reset` / :meth:`gymnasium.Env.step`.
   * - Output
     - A set of atomic predicate strings (:math:`L(obs) \subseteq \mathcal{AP}`).
   * - Empty output
     - Valid (no predicates satisfied).
   * - Determinism
     - Strongly recommended.

Labelled Environment Wrapper
----------------------------

To standardise access to labels, MASA provides :class:`masa.common.labelled_env.LabelledEnv`, a lightweight
Gymnasium wrapper that computes labels on **every** :meth:`~gymnasium.Env.reset` and :meth:`~gymnasium.Env.step`
and injects them into the ``info`` dictionary under the key ``"labels"``:

.. code-block:: python

   info["labels"] = set(label_fn(obs))

Usage
~~~~~

.. code-block:: python

   import gymnasium as gym
   from masa.common.labelled_env import LabelledEnv

   env = gym.make("CartPole-v1")
   env = LabelledEnv(env, label_fn)

   obs, info = env.reset()
   labels = info["labels"]

   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
   labels = info["labels"]

API Reference
~~~~~~~~~~~~~

.. autoclass:: masa.common.labelled_env.LabelledEnv
   :members:
   :show-inheritance:

Common Pitfalls
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Pitfall
     - Recommendation
   * - Returning non-strings
     - Always return strings naming atomic predicates.
   * - Using environment internals
     - Derive labels from observations (not hidden state).
   * - Stateful label functions
     - Prefer pure, stateless functions (state belongs in constraints/monitors).
   * - Inconsistent predicate vocabulary
     - Define and document :math:`\mathcal{AP}` clearly (including spelling/casing).
