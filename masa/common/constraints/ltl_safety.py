"""
Overview
--------

LTL-style safety constraints via DFA monitoring and product constructions.

This module supports *safety constraints* expressed using a deterministic finite
automaton (DFA) derived from an LTL (or LTL-like) specification.

High-level idea
---------------
Given a labelled MDP and a DFA over the same atomic propositions, we can build a
*product MDP* whose states track both the base state and the current automaton
state. If the DFA has an "unsafe" accepting set :math:`F`, then safety can be
monitored by checking whether the automaton enters :math:`F`.

Let:

- base MDP have :math:`n` states and :math:`m` actions,
- DFA have :math:`k` automaton states.

Then the product state space has size :math:`n \\cdot k`.

Product transitions
-------------------
If the base transition kernel is :math:`P(s'\\mid s,a)` and the DFA transition
function is :math:`\\delta(q, L(s))`, then the product transition is:

.. math::

   P_\\otimes((s', q') \\mid (s,q), a)
   = P(s'\\mid s,a) \\cdot \\mathbf{1}\\{ q' = \\delta(q, L(s)) \\}.

The helper functions in this module create either a dense transition tensor or
a sparse successor representation for the product.

Safety cost
-----------
The monitor uses a cost function derived from the DFA. A common convention is:

.. math::

   c_t = \\begin{cases}
     1 & \\text{if } q_t \\in F \\\\
     0 & \\text{otherwise}
   \\end{cases}

and the episode is safe iff the total number of unsafe visits is zero.

Notes
-----
- The concrete cost construction is delegated to :func:`masa.common.ltl.dfa_to_costfn`.
- :class:`LTLSafetyEnv` also augments observations to include the automaton state,
  enabling *product-state* learning in model-free settings.

API Reference
-------------
"""

from __future__ import annotations
from typing import Any, Dict, List, Iterable, Callable, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from masa.common.label_fn import LabelFn
from masa.common.constraints.base import Constraint, BaseConstraintEnv
from masa.common.ltl import DFA, dfa_to_costfn
from masa.common.dummy import make_dfa as make_dummy_dfa

State = int
Action = int
ProdState = int

product_cost_fn = lambda labels: 1.0 if "accepting" in labels else 0.0

def create_product_transition_matrix(
    n_states: int,
    n_actions: int,
    transition_matrix: np.ndarray, 
    dfa: DFA,
    label_fn: LabelFn,
) -> np.ndarray:
    """Create the dense product transition tensor for (MDP × DFA).

    Given a dense base transition tensor of shape ``(n_states, n_states, n_actions)``,
    this constructs the corresponding dense product transition tensor of shape
    ``(n_states * n_aut, n_states * n_aut, n_actions)``.

    The DFA transition is computed from labels of the *current* base state ``s``
    (i.e., it applies :math:`q' = \\delta(q, L(s))`), which matches the product
    formulation used in the code:

    .. math::

       P_\\otimes((s', q') \\mid (s,q), a)
       = P(s'\\mid s,a) \\cdot \\mathbf{1}\\{ q' = \\delta(q, L(s)) \\}.

    Args:
        n_states: Number of base MDP states.
        n_actions: Number of actions.
        transition_matrix: Dense base transition tensor with shape
            ``(n_states, n_states, n_actions)``, where
            ``transition_matrix[s, s_next, a] = P(s_next | s, a)``.
        dfa: Deterministic finite automaton.
        label_fn: Labelling function ``L(s) -> set[str]``.

    Returns:
        Dense product transition tensor with shape
        ``(n_states * n_aut, n_states * n_aut, n_actions)``.

    Raises:
        AssertionError: If the provided transition matrix does not have the expected shape.

    """

    assert len(transition_matrix.shape) == 3 and transition_matrix.shape[0] == transition_matrix.shape[1], \
    f"Expected transition matrix with shape (n_states, n_states, n_actions), got shape {transition_matrix.shape} instead"

    aut_states = list(dfa.states)
    n_aut = len(aut_states)
    aut_index = {q: i for i, q in enumerate(aut_states)}

    assert n_states == transition_matrix.shape[0], \
    "Something went wrong, the provided n_states does not equal the number of states in transition_matrix"
    f"Got n_states = {n_states} and transition_matrix.shape[0] == {transition_matrix.shape[0]}"

    assert n_actions == transition_matrix.shape[2], \
    "Something went wrong, the provided n_actions does not equal the number of states in transition_matrix"
    f"Got n_actions = {n_actions} and transition_matrix.shape[2] == {transition_matrix.shape[2]}"

    sat = np.zeros((n_aut, n_aut, n_states), dtype=np.float32)

    for i, i_state in enumerate(aut_states):
        for j, j_state in enumerate(aut_states):
            if i == j:
                continue
            if dfa.has_edge(i_state, j_state):
                edge = dfa.edges[i_state][j_state]
                # i_j_sat_relation[s] = edge.sat(label_fn(s))
                i_j_sat_relation = np.array(
                    [edge.sat(label_fn(s)) for s in range(n_states)],
                    dtype=np.float32,
                )
                sat[i, j, :] = i_j_sat_relation

    sat_no_diag = sat.copy()
    idx = np.arange(n_aut)
    sat_no_diag[idx, idx, :] = 0.0

    # Default behvaiour: states with no outgoing edge loop in the automata
    outgoing_any = sat_no_diag.max(axis=1)
    loop_sat = 1.0 - outgoing_any
    sat[idx, idx, :] = loop_sat

    product = np.einsum('ijs,ska->jsika', sat, transition_matrix.astype(np.float32))

    n_prod_states = n_states * n_aut
    product_transition_matrix = product.reshape(
        n_aut * n_states,
        n_aut * n_states,
        n_actions,
    )

    return product_transition_matrix

def create_product_successor_states_and_probabilities(
    n_states: int,
    n_actions: int,
    successor_states: Dict[State, List[State]],
    probabilities: Dict[Tuple[State, Action], np.ndarray],
    dfa: DFA,
    label_fn: LabelFn,
) -> Tuple[Dict[ProdState, List[ProdState]], Dict[Tuple[ProdState, Action], np.ndarray]]:
    """Create a sparse product successor representation (MDP x DFA).

    This constructs:

    - ``prod_successor_states``: mapping ``prod_state -> list[prod_state_next]``
    - ``prod_probabilities``: mapping ``(prod_state, action) -> probs``

    where probability vectors are copied from the base representation and the
    automaton transition is determined by the current base state labels.

    Product state indexing
    ~~~~~~~~~~~~~~~~~~~~~~
    The code uses the encoding:

    .. math::

       (q\\_\\text{idx}, s) \\mapsto \\text{prod} = q\\_\\text{idx} \\cdot n\\_\\text{states} + s.

    Args:
        n_states: Number of base MDP states.
        n_actions: Number of actions.
        successor_states: Mapping ``s -> [s_1, s_2, ...]`` listing successors of ``s``.
        probabilities: Mapping ``(s, a) -> p`` where ``p`` is a 1-D array aligned with ``successor_states[s]`` and sums to 1.
        dfa: Deterministic finite automaton.
        label_fn: Labelling function ``L(s) -> set[str]``.

    Returns:
        A tuple ``(prod_successor_states, prod_probabilities)`` representing the
        product dynamics.

    Raises:
        AssertionError: If ``n_states`` is inconsistent with the keys in ``successor_states``.

    """

    base_states = sorted(successor_states.keys())

    assert n_states == len(base_states), \
    "Something went wrong, the provided n_states does not equal the numebr of states in successor_states "
    f"Got n_states = {n_states} and len(successor_states) = {len(base_states)}"

    state_index = {s: idx for idx, s in enumerate(base_states)}
    
    aut_states = list(dfa.states)
    n_aut = len(aut_states)
    aut_index = {q: i for i, q in enumerate(aut_states)}

    next_aut = np.zeros((n_aut, n_states), dtype=np.int64)

    for q_idx, q in enumerate(aut_states):
        for s in base_states:
            labels = label_fn(s)
            q_next = dfa.transition(q, labels)
            j_idx = aut_index[q_next]
            next_aut[q_idx, s] = j_idx

    prod_successor_states: Dict[ProdState, List[ProdState]] = {}
    prod_probabilities: Dict[Tuple[ProdState, Action], np.ndarray] = {}

    for q_idx, q in enumerate(aut_states):
        for s in base_states:
            prod_state = q_idx * n_states + s

            succ_s = successor_states.get(s, [])
            if not succ_s:
                continue

            j_idx = next_aut[q_idx, s]

            prod_succ_list = [j_idx * n_states + s_prime for s_prime in succ_s]
            prod_successor_states[prod_state] = prod_succ_list

            for a in range(n_actions):
                probs_sa = probabilities.get((s, a))
                if probs_sa is None:
                    continue

                prod_probabilities[(prod_state, a)] = probs_sa.copy()

    return prod_successor_states, prod_probabilities

def create_product_label_fn(
    n_states: int,
    dfa: DFA,
) -> Callable[[ProdState], Set[str]]:
    """Create a label function on product states indicating DFA acceptance.

    The returned labelling function maps a product-state index to ``{"accepting"}``
    if the embedded DFA state is accepting, and to the empty set otherwise.

    Args:
        n_states: Number of base states used in product encoding.
        dfa: DFA defining which automaton indices are accepting.

    Returns:
        A callable ``L_prod(prod_state) -> set[str]`` suitable for cost functions
        such as::

            cost = 1.0 if "accepting" in labels else 0.0

    """

    aut_states = list(dfa.states)
    aut_index = {q: i for i, q in enumerate(aut_states)}
    accepting_indexes = {aut_index[q] for q in dfa.accepting}

    def product_label_fn(obs):
        aut_state_idx = obs // n_states
        if aut_state_idx in accepting_indexes:
            return {"accepting"}
        else:
            return set()

    return product_label_fn

class LTLSafety(Constraint):
    """DFA-based safety monitor.

    This monitor uses :func:`masa.common.ltl.dfa_to_costfn` to obtain a stateful
    cost function that:

    - tracks the current DFA state,
    - returns a scalar cost indicating safety violation.

    A common convention is binary step cost:

    .. math::

       c_t \\in \\{0, 1\\}, \\quad
       c_t = 1 \\iff \\text{DFA enters/indicates an unsafe accepting condition}.

    The episode is considered satisfied iff no unsafe event occurs:

    .. math::

       \\text{satisfied} \\iff \\sum_t \\mathbf{1}[c_t \\ge 0.5] = 0.

    Args:
        dfa: DFA describing the safety property.

    Attributes:
        cost_fn: Stateful cost object derived from the DFA (exposes DFA state).
        safe: Boolean flag tracking whether any violation has occurred.
        step_cost: Most recent step cost.
        total_unsafe: Count of unsafe steps (as floats, per current code).

    """

    def __init__(self, dfa: DFA):
        self.cost_fn = dfa_to_costfn(dfa)

    def reset(self):
        """Reset the safety monitor and underlying DFA-cost state."""
        self.safe = True
        self.step_cost = 0.0
        self.total_unsafe = 0.0
        self.cost_fn.reset()

    def update(self, labels: Iterable[str]):
        """Update the DFA-cost state and safety flags.

        Args:
            labels: Iterable of atomic propositions true at the current step.
        """
        self.step_cost = self.cost_fn(labels)
        self.total_unsafe += float(self.step_cost >= 0.5)
        self.safe = self.safe and (not self.total_unsafe)

    def get_automaton_state(self):
        """Return the current DFA state from the underlying DFA-cost object."""
        return self.cost_fn.automaton_state

    def get_dfa(self):
        """Return the DFA used by the underlying DFA-cost object."""
        return self.cost_fn.dfa

    def satisfied(self) -> bool:
        """Whether the episode remains safe so far."""
        return self.safe

    def episode_metric(self) -> Dict[str, float]:
        """End-of-episode metrics.

        Returns:
            Dict containing:

            - ``"cum_unsafe"``: count of unsafe steps,
            - ``"satisfied"``: 1.0 if safe else 0.0.
        """
        return {"cum_unsafe": float(self.total_unsafe), "satisfied": float(self.satisfied())}

    def step_metric(self) -> Dict[str, float]:
        """Per-step metrics.

        Returns:
            Dict containing:

            - ``"cost"``: current step cost,
            - ``"violation"``: 1.0 if ``cost >= 0.5`` else 0.0.
        """
        return {"cost": self.step_cost, "violation": float(self.step_cost >= 0.5)}

    @property
    def constraint_type(self) -> str:
        """Stable identifier string: ``"ltl_safety"``."""
        return "ltl_safety"


class LTLSafetyEnv(BaseConstraintEnv):
    """Gymnasium wrapper that monitors LTL safety and augments observations.

    This wrapper attaches :class:`LTLSafety` to the environment and augments the
    observation space to include the current DFA state, enabling model-free
    learning over the *product*.

    The *representation* of the product observation is controlled by ``obs_type``:

    - ``obs_type="discrete"``:
      Requires the underlying observation space to be :class:`gymnasium.spaces.Discrete`.
      The observation becomes a single discrete index encoding both
      base state and automaton state:

      .. math::

         \\text{obs}_\\otimes = q\\_\\text{idx} \\cdot n + s.

    - ``obs_type="box"``:
      Produces a :class:`gymnasium.spaces.Box` observation by concatenating a
      one-hot encoding of the automaton state.

      * If the base space is :class:`~gymnasium.spaces.Box` (1-D only), the result is
        ``concat([obs, one_hot(q)])``.
      * If the base space is :class:`~gymnasium.spaces.Discrete`, the result is
        ``concat([one_hot(s), one_hot(q)])``.

    - ``obs_type="dict"``:
      Produces a :class:`gymnasium.spaces.Dict` observation with keys:
      ``"orig"`` and ``"automaton"``.
      ``"automaton"`` is always Discrete; ``"orig"`` matches the original observation
      (for :class:`~gymnasium.spaces.Discrete`) or the original vector (for 1-D
      :class:`~gymnasium.spaces.Box`). For original :class:`~gymnasium.spaces.Dict`,
      this wrapper adds an ``"automaton"`` key to the existing dict.

    The wrapper also writes ``info["automaton_state"]`` each step/reset.

    Args:
        env: Base environment (must be a :class:`~masa.common.labelled_env.LabelledEnv`).
        dfa: DFA for safety monitoring. Defaults to a dummy DFA.
        obs_type: One of ``{"discrete", "box", "dict"}``, controlling the product
            observation representation.
        **kw: Extra keyword arguments forwarded to :class:`BaseConstraintEnv`.

    Raises:
        ValueError: If ``dfa.num_automaton_states`` is non-positive.
        ValueError: If ``obs_type`` is not in ``{"discrete", "box", "dict"}``.
        TypeError: If an incompatible configuration is requested (e.g.
            ``obs_type="discrete"`` but the base observation space is not Discrete),
            or if the base space is unsupported.
        TypeError: If ``obs_type`` requires a 1-D Box but the Box is not 1-D.
    """

    def __init__(
        self,
        env: gym.Env,
        dfa: "DFA" = make_dummy_dfa(),
        obs_type: str = "discrete",
        **kw: Any,
    ):
        if obs_type not in ("discrete", "box", "dict"):
            raise ValueError(
                f"obs_type must be one of ['discrete', 'box', 'dict'], got {obs_type!r}"
            )
            
        super().__init__(env, LTLSafety(dfa=dfa), **kw)
        self._num_automaton_states = int(dfa.num_automaton_states)
        if self._num_automaton_states < 1:
            raise ValueError("dfa.num_automaton_states must be non-zero and positive")

        self._automaton_states_idx = {q: i for i, q in enumerate(dfa.states)}

        self._orig_obs_space = env.observation_space
        self._obs_type = obs_type
        self._box_dtype = np.float32

        self.observation_space = self._make_augmented_obs_space(self._orig_obs_space, self._obs_type)

    def _make_augmented_obs_space(self, orig: spaces.Space, obs_type: str) -> spaces.Space:
        """Construct the augmented observation space.

        Args:
            orig: Original observation space of the wrapped environment.

        Returns:
            A new observation space that includes the automaton state.

        Raises:
            TypeError: If the observation space type is unsupported, or if a Box
                space is not 1-D.
        """
        if isinstance(orig, spaces.Discrete):
            n = int(orig.n)
            if obs_type == "discrete":
                return spaces.Discrete(n * self._num_automaton_states)
            if obs_type == "box":
                dim = n + self._num_automaton_states
                return spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(dim,),
                    dtype=self._box_dtype,
                )
            if obs_type == "dict":
                return spaces.Dict(
                    {
                        "orig": spaces.Discrete(n),
                        "automaton": spaces.Discrete(self._num_automaton_states)
                    }
                )
            raise RuntimeError(f"Unhandled obs_type: {obs_type!r}")

        if isinstance(orig, spaces.Box):
            if obs_type == "discrete":
                raise TypeError(
                    "Incompatible configuration: obs_type='discrete' requires a Discrete "
                    "base observation space, but got Box."
                )
            if orig.shape is None or len(orig.shape) != 1:
                raise TypeError(
                    f"LTLSafetyEnv only supports 1-D Box for augmentation; got shape {orig.shape}"
                )
            d = int(orig.shape[0])
            if obs_type == "box":
                low = np.concatenate(
                    [
                        orig.low.astype(self._box_dtype, copy=False),
                        np.zeros(self._num_automaton_states, dtype=self._box_dtype),
                    ]
                )
                high = np.concatenate(
                    [
                        orig.high.astype(self._box_dtype, copy=False),
                        np.ones(self._num_automaton_states, dtype=self._box_dtype),
                    ]
                )
                return spaces.Box(low=low, high=high, dtype=self._box_dtype)
            if obs_type == "dict":
                return spaces.Dict(
                    {
                        "orig": orig,
                        "automaton": spaces.Discrete(self._num_automaton_states)
                    }
                )
            raise RuntimeError(f"Unhandled obs_type: {obs_type!r}")

        if isinstance(orig, spaces.Dict):
            if obs_type == "discrete":
                raise TypeError(
                    "Incompatible configuration: obs_type='discrete' requires a Discrete "
                    "base observation space, but got Dict."
                )
            if obs_type == "box":
                raise TypeError(
                    "Incompatible configuration: obs_type='box' is not supported when the "
                    "base observation space is Dict (cannot flatten generically). "
                    "Use obs_type='dict' instead."
                )
            new_spaces = dict(orig.spaces)
            new_spaces["automaton"] = spaces.Discrete(self._num_automaton_states)
            return spaces.Dict(new_spaces)

        raise TypeError(
            f"LTLSafetyEnv does not support base observation space type {type(orig).__name__}. "
            "Supported base spaces: Discrete, 1-D Box, Dict."
        )

    def _one_hot(self, idx: int, dim: int) -> np.ndarray:
        """One-hot encode an index into a vector of length ``dim``.

        Args:
            idx: index to encomde.
            dim: length of one-hot encoding.

        Returns:
            A 1-D numpy array of shape ``(dim,)`` containing a
            one-hot encoding. If ``idx`` is out of range, returns the all-zeros vector.
        """
        enc = np.zeros(dim, dtype=self._box_dtype)
        if 0 <= int(idx) < dim:
            enc[int(idx)] = 1.0
        return enc

    def _augment_obs(self, obs: Any) -> Any:
        """Augment a base observation with the current automaton state.

        Args:
            obs: Base observation returned by the wrapped environment.

        Returns:
            Augmented observation matching :attr:`observation_space`.

        Raises:
            TypeError: If the base observation does not match the expected type/shape
                implied by the observation space.
            RuntimeError: If the wrapper is in an unexpected observation-space state.
        """
        q_state = self._constraint.get_automaton_state()
        q_idx = int(self._automaton_states_idx[q_state])

        orig = self._orig_obs_space
        obs_type = self._obs_type

        if isinstance(orig, spaces.Discrete):
            if not isinstance(obs, (int, np.integer)):
                raise TypeError(f"Expected Discrete obs as int, got {type(obs).__name__}")
            s = int(obs)
            if not (0 <= s < int(orig.n)):
                raise TypeError(f"Discrete obs out of range: got {s}, expected [0, {orig.n})")
            if obs_type == "discrete":
                return int(orig.n) * q_idx + s
            if obs_type == "box":
                return np.concatenate(
                    [self._one_hot(s, int(orig.n)), self._one_hot(q_idx, self._num_automaton_states)],
                    axis=0,
                )
            if obs_type == "dict":
                return {
                    "orig": s,
                    "automaton": q_idx,
                }
            raise RuntimeError(f"Unhandled obs_type: {obs_type!r}")
        if isinstance(orig, spaces.Box):
            if obs_type == "discrete":
                raise RuntimeError("obs_type='discrete' with Box base should have been rejected.")
            arr = obs if isinstance(obs, np.ndarray) else np.asarray(obs, dtype=self._box_dtype)
            if arr.ndim != 1:
                raise TypeError(
                    f"Expected 1-D Box observation, got shape {getattr(arr, 'shape', None)}"
                )
            arr = arr.astype(self._box_dtype, copy=False)
            if obs_type == "box":
                return np.concatenate([arr, self._one_hot(q_idx, self._num_automaton_states)], axis=0)
            if obs_type == "dict":
                return {
                    "orig": arr,
                    "automaton": q_idx
                }
            raise RuntimeError(f"Unhandled obs_type: {obs_type!r}")
        if isinstance(orig, spaces.Dict):
            if obs_type != "dict":
                raise RuntimeError("Only obs_type='dict' is supported for Dict base spaces.")
            if not isinstance(obs, dict):
                raise TypeError(f"Expected Dict obs as dict, got {type(obs).__name__}")
            out = dict(obs)
            out["automaton"] = q_idx
            return out

        raise RuntimeError(f"Unexpected base observation space type {type(orig).__name__}")

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._constraint.reset()
        labels = info.get("labels", set())
        self._constraint.update(labels)
        info['automaton_state'] = self._constraint.get_automaton_state()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        labels = info.get("labels", set())
        self._constraint.update(labels)
        info['automaton_state'] = self._constraint.get_automaton_state()
        return self._augment_obs(obs), reward, terminated, truncated, info
    

        