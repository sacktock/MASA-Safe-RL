from masa.common.label_fn import LabelFn
from typing import Any, List, Dict, Union, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

jax.config.update("jax_enable_x64", True)

DenseKernel = np.ndarray
CompactKernel = Tuple[np.ndarray, np.ndarray]
Kernel = Union[DenseKernel, CompactKernel]

def kernel_n_states(kernel: Kernel) -> int:
    """Return the number of states induced by a transition kernel.

    This helper supports both:
      * **Dense kernels**: a 2D Markov-chain transition matrix with shape
        ``(n_states, n_states)``, where column ``s`` is the distribution over
        next states from state ``s``.
      * **Compact kernels**: a tuple ``(succ, p)`` where:
          - ``succ`` has shape ``(K, n_states)`` and stores the integer ids of up to
            ``K`` successor states per state.
          - ``p`` has shape ``(K, n_states)`` and stores the corresponding
            probabilities for each successor.

    Args:
        kernel: Dense or compact kernel representation.

    Returns:
        The number of states ``n_states``.
    """
    if isinstance(kernel, tuple):
        succ, p = kernel
        return succ.shape[1]
    return kernel.shape[1]

class BoundedPCTLFormula:
    """Base class for bounded PCTL-style formulas.

    The formula API is centered around two related operations:

      * ``sat(...)``: evaluate the formula at the **final bound** and return a
        per-state satisfaction indicator (as ``float64`` in ``{0.0, 1.0}``).
      * ``_prob_seq(...)``: compute a *sequence* of satisfaction/probability values
        up to some horizon ``max_k`` (inclusive), returning an array of shape
        ``(max_k + 1, n_states)``.

    Notes:
        - This module uses a vectorized labeling representation ``vec_label_fn``
          of shape ``(n_atoms, n_states)`` where ``vec_label_fn[i, s]`` is 1.0 if
          atomic predicate ``i`` holds in state ``s`` and 0.0 otherwise.
        - The evaluation is performed on a **Markov chain** kernel (policy has
          already been applied if coming from an MDP).
        - Implementations typically use JAX for accelerated scans and vmap.

    Attributes:
        bound: Total time bound of the formula, including any nested subformula
            bounds contributed by probabilistic operators.
    """

    def __init__(self) -> None:
        pass

    @property
    def _bound(self) -> int:
        """Internal bound implementation for subclasses.

        Subclasses must implement this property and return the total bound for
        the formula instance.

        Returns:
            Integer bound (>= 0).

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    def bound(self) -> int:
        """Total bound of this formula (read-only)."""
        return self._bound

    def _prob_seq(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
        max_k: int | None = None,
    ) -> np.ndarray:
        """Compute a satisfaction/probability sequence up to horizon ``max_k``.

        Default behavior treats the formula as a *state formula* and simply
        repeats the time-0 satisfaction vector across time steps.

        Args:
            kernel: Transition kernel (dense or compact) of the Markov chain.
            vec_label_fn: Vectorized labeling matrix of shape ``(n_atoms, n_states)``.
            atom_dict: Mapping from atomic predicate name to row index in
                ``vec_label_fn``.
            max_k: Optional horizon. If ``None``, defaults to ``self.bound``.

        Returns:
            Array of shape ``(max_k + 1, n_states)``.

        Notes:
            Subclasses implementing temporal/probabilistic operators should
            override this method.
        """
        if max_k is None:
            max_k = self.bound

        v = self.sat(kernel, vec_label_fn, atom_dict)
        return np.repeat(v[None, :], max_k + 1, axis=0)

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Evaluate the formula at its bound for all states.

        Args:
            kernel: Transition kernel (dense or compact) of the Markov chain.
            vec_label_fn: Vectorized labeling matrix of shape ``(n_atoms, n_states)``.
            atom_dict: Mapping from atomic predicate name to row index in
                ``vec_label_fn``.

        Returns:
            A float64 array of shape ``(n_states,)`` with values in ``{0.0, 1.0}``.
        """
        return self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound)[
            self.bound
        ]

class Next(BoundedPCTLFormula):
    r"""Bounded PCTL :math:`X` (next) operator with a probability threshold.

    Semantics (informal):
        :math:`\mathbb{P}_{\geq p}[ X \Phi ]` holds in state :math:`s` iff the probability that :math:`\Phi` holds
        in the next state is at least :math:`p`.

    This implementation computes the probability sequence for :math:`X \Phi` via one-step
    expectation under the Markov chain kernel.

    Args:
        prob: Probability threshold in ``[0, 1]``.
        subformula: Subformula :math:`\Phi` evaluated at the next state.

    Attributes:
        prob: Probability threshold.
        subformula: The nested formula :math:`\Phi`.
        bound_param: The local horizon contributed by this operator (fixed to 1).
    """

    def __init__(self, prob: float, subformula: BoundedPCTLFormula):
        super().__init__()
        self.prob = prob
        self.subformula = subformula
        self.bound_param = 1

    @property
    def _bound(self):
        """Total bound including nested subformula contribution."""
        return self.bound_param + self.subformula.bound

    @staticmethod
    @jit
    def _next_prob_seq_core_dense(m: jnp.ndarray, sub_seq: jnp.ndarray) -> jnp.ndarray:
        """JAX core for next-probability sequences using a dense kernel.

        Args:
            m: Dense transition matrix with shape ``(n_states, n_states)``. Column
                ``s`` is the distribution over successors of ``s``.
            sub_seq: Sequence for the subformula of shape ``(T, n_states)``.

        Returns:
            Sequence for ``X subformula`` of shape ``(T + 1, n_states)``, with the
            first row set to zeros (probability at step 0 is defined as 0 for
            the shifted operator).
        """
        tail = (m.T @ sub_seq.T)
        tail = jnp.swapaxes(tail, 0, 1)
        zeros = jnp.zeros_like(tail[:1, :])
        return jnp.concatenate([zeros, tail], axis=0)

    @staticmethod
    @partial(jit, static_argnames=("K",))
    def _next_prob_seq_core_compact(
        succ: jnp.ndarray, p: jnp.ndarray, sub_seq: jnp.ndarray, K: int
    ) -> jnp.ndarray:
        """JAX core for next-probability sequences using a compact kernel.

        Args:
            succ: Successor index matrix of shape ``(K, n_states)``.
            p: Probabilities aligned with ``succ``, shape ``(K, n_states)``.
            sub_seq: Sequence for the subformula of shape ``(T, n_states)``.
            K: Maximum number of successors per state (static for JIT).

        Returns:
            Sequence for ``X subformula`` of shape ``(T + 1, n_states)``, with the
            first row set to zeros.
        """
        def one_step(v):
            v_succ = v[succ]
            return jnp.sum(p * v_succ, axis=0)

        tail = jax.vmap(one_step)(sub_seq)
        zeros = jnp.zeros_like(tail[:1, :])
        return jnp.concatenate([zeros, tail], axis=0)

    def _prob_seq(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
        max_k: int | None = None,
    ) -> np.ndarray:
        """Compute probability sequence for the ``Next`` operator.

        Args:
            kernel: Markov-chain kernel (dense ``(S,S)`` or compact ``(succ,p)``).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, n_states)``.
            atom_dict: Mapping from atom name to ``vec_label_fn`` row.
            max_k: Sequence horizon for this operator. If ``None``, defaults to the
                local bound ``bound_param`` (i.e., 1).

        Returns:
            Array of shape ``(max_k + 1, n_states)`` containing probabilities that
            the subformula holds at the shifted time index.

        Notes:
            If ``max_k == 0``, this returns a single row of zeros.
        """
        if max_k is None:
            max_k = self.bound_param

        n_states = kernel_n_states(kernel)
        if max_k == 0:
            return np.zeros((1, n_states), dtype=np.float64)

        sub_seq_np = self.subformula._prob_seq(kernel, vec_label_fn, atom_dict, max_k=max_k - 1)
        sub_seq_j = jnp.asarray(sub_seq_np, dtype=jnp.float64)

        if isinstance(kernel, tuple):
            succ, p = kernel
            succ_j = jnp.asarray(succ, dtype=jnp.int64)
            p_j = jnp.asarray(p, dtype=jnp.float64)
            seq_j = self._next_prob_seq_core_compact(succ_j, p_j, sub_seq_j, K=succ.shape[0])
        else:
            m_j = jnp.asarray(kernel, dtype=jnp.float64)
            seq_j = self._next_prob_seq_core_dense(m_j, sub_seq_j)

        return np.asarray(seq_j, dtype=np.float64)

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        r"""Return satisfaction indicator for :math:`\mathbb{P}_{\geq p} [ X \Phi ]` at the bound.

        Args:
            kernel: Markov-chain kernel (dense or compact).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, n_states)``.
            atom_dict: Mapping from atom name to ``vec_label_fn`` row.

        Returns:
            Float64 array ``(n_states,)`` in ``{0.0, 1.0}`` indicating whether the
            probability meets the threshold.
        """
        probs = self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound_param)[
            self.bound_param
        ]
        return (probs >= self.prob).astype(np.float64)

class Until(BoundedPCTLFormula):
    r"""Bounded PCTL :math:`U` (until) operator with a probability threshold.

    Semantics (informal):
        :math:`\mathbb{P}_{\geq p} [ \Phi_1 U^{\leq B} \Phi_2 ]` holds in state :math:`s` iff the probability that
        :math:`\Phi_2` becomes true within :math:`B` steps while :math:`\Phi_1` holds at all preceding
        steps is at least :math:`p`.

    This implementation computes a sequence :math:`P_k(s)` for :math:`k=0 \ldots B` using the
    standard bounded-until recurrence:
        - :math:`P_0 = sat2`
        - :math:`P_{k+1} = sat2 + ((1 - sat2) * sat1) * \mathbb{E}[P_k(next_state)]`

    Args:
        prob: Probability threshold in ``[0, 1]``.
        bound: Local bound :math:`B` (number of steps).
        subformula_1: :math:`\Phi_1` (continuation condition).
        subformula_2: :math:`\Phi_2` (target condition).

    Attributes:
        prob: Probability threshold.
        bound_param: Local bound :math:`B`.
        subformula_1: Continuation subformula.
        subformula_2: Target subformula.
    """

    def __init__(self, prob: float, bound, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.prob = prob
        self.bound_param = int(bound)
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        """Total bound including nested subformula contributions."""
        return self.bound_param + self.subformula_1.bound + self.subformula_2.bound

    @staticmethod
    @partial(jit, static_argnames=("max_k",))
    def _until_prob_seq_core_dense(
        m: jnp.ndarray, sat1: jnp.ndarray, sat2: jnp.ndarray, max_k: int
    ) -> jnp.ndarray:
        r"""JAX core for bounded-until probability sequences with a dense kernel.

        Args:
            m: Dense transition matrix of shape ``(n_states, n_states)``.
            sat1: Satisfaction mask for :math:`\Phi_1`, shape ``(n_states,)`` in ``{0,1}``.
            sat2: Satisfaction mask for :math:`\Phi_2`, shape ``(n_states,)`` in ``{0,1}``.
            max_k: Bound :math:`B` (number of recurrence steps).

        Returns:
            Probability sequence of shape ``(max_k + 1, n_states)``.
        """
        cont_mask = (1.0 - sat2) * sat1
        prob0 = sat2

        def body(prob, _):
            exp = m.T @ prob
            next_prob = sat2 + cont_mask * exp
            return next_prob, next_prob

        _, tail = jax.lax.scan(body, prob0, None, length=max_k)
        return jnp.concatenate([prob0[None, :], tail], axis=0)

    @staticmethod
    @partial(jit, static_argnames=("max_k", "K"))
    def _until_prob_seq_core_compact(
        succ: jnp.ndarray, p: jnp.ndarray, sat1: jnp.ndarray, sat2: jnp.ndarray, max_k: int, K: int
    ) -> jnp.ndarray:
        """JAX core for bounded-until sequences with a compact kernel.

        Args:
            succ: Successor matrix of shape ``(K, n_states)``.
            p: Successor probabilities of shape ``(K, n_states)``.
            sat1: Satisfaction mask for :math:`\Phi_1`, shape ``(n_states,)`` in ``{0,1}``.
            sat2: Satisfaction mask for :math:`\Phi_2`, shape ``(n_states,)`` in ``{0,1}``.
            max_k: Bound :math:`B` (number of recurrence steps).
            K: Max successors per state (static for JIT).

        Returns:
            Probability sequence of shape ``(max_k + 1, n_states)``.
        """
        cont_mask = (1.0 - sat2) * sat1
        prob0 = sat2

        def exp_step(v):
            v_succ = v[succ]
            return jnp.sum(p * v_succ, axis=0)

        def body(prob, _):
            exp = exp_step(prob)
            next_prob = sat2 + cont_mask * exp
            return next_prob, next_prob

        _, tail = jax.lax.scan(body, prob0, None, length=max_k)
        return jnp.concatenate([prob0[None, :], tail], axis=0)

    def _prob_seq(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
        max_k: int | None = None,
    ) -> np.ndarray:
        """Compute probability sequence for the bounded ``Until`` operator.

        Args:
            kernel: Markov-chain kernel (dense ``(S,S)`` or compact ``(succ,p)``).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, n_states)``.
            atom_dict: Mapping from atom name to ``vec_label_fn`` row.
            max_k: Local bound to compute up to. If ``None``, defaults to
                ``self.bound_param``.

        Returns:
            Array of shape ``(max_k + 1, n_states)`` containing bounded-until
            satisfaction probabilities.
        """
        if max_k is None:
            max_k = self.bound_param

        sat1_np = self.subformula_1.sat(kernel, vec_label_fn, atom_dict).astype(np.float64)
        sat2_np = self.subformula_2.sat(kernel, vec_label_fn, atom_dict).astype(np.float64)

        sat1_j = jnp.asarray(sat1_np, dtype=jnp.float64)
        sat2_j = jnp.asarray(sat2_np, dtype=jnp.float64)

        if isinstance(kernel, tuple):
            succ, p = kernel
            succ_j = jnp.asarray(succ, dtype=jnp.int64)
            p_j = jnp.asarray(p, dtype=jnp.float64)
            seq_j = self._until_prob_seq_core_compact(succ_j, p_j, sat1_j, sat2_j, max_k=max_k, K=succ.shape[0])
        else:
            m_j = jnp.asarray(kernel, dtype=jnp.float64)
            seq_j = self._until_prob_seq_core_dense(m_j, sat1_j, sat2_j, max_k=max_k)

        return np.asarray(seq_j, dtype=np.float64)

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        r"""Return satisfaction indicator for :math:`\mathbb{P}_{\geq p}[ \Phi_1 U^{\leq B} \Phi_2 ]`.

        Args:
            kernel: Markov-chain kernel (dense or compact).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, n_states)``.
            atom_dict: Mapping from atom name to ``vec_label_fn`` row.

        Returns:
            Float64 array ``(n_states,)`` in ``{0.0, 1.0}`` indicating whether the
            bounded-until probability meets the threshold.
        """
        probs = self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound_param)[
            self.bound_param
        ]
        return (probs >= self.prob).astype(np.float64)

class Always(BoundedPCTLFormula):
    r"""Bounded PCTL :math:`G` (always) operator with a probability threshold.

    Semantics (informal):
        :math:`\mathbb{P}_{\geq p}[ G^{\leq B} \Phi ]` holds if, within the bound, :math:`\Phi` holds at all steps with
        probability at least :math:`p`.

    This is implemented via duality:
        :math:`G^{\leq B} \Phi` is equivalent to :math:`\neg ( \top U^{\leq B} \neg \Phi )`,
    with an appropriate threshold transformation.

    Args:
        prob: Probability threshold in ``[0, 1]``.
        bound: Local bound :math:`B`.
        subformula: Subformula :math:`\Phi`.

    Attributes:
        prob: Probability threshold.
        bound_param: Local bound :math:`B`.
        subformula: Subformula :math:`\Phi`.
    """

    def __init__(self, prob: float, bound: int, subformula: BoundedPCTLFormula):
        super().__init__()
        self.prob = float(prob)
        self.bound_param = int(bound)
        self.subformula = subformula

        self._inner = Neg(
            Until(
                prob=1.0-self.prob,
                bound=self.bound_param,
                subformula_1=Truth(),
                subformula_2=Neg(self.subformula),
            )
        )

    @property
    def _bound(self) -> int:
        """Total bound delegated to the desugared inner formula."""
        return self._inner.bound

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        """Delegate sequence computation to the desugared inner formula."""
        return self._inner._prob_seq(kernel, vec_label_fn, atom_dict, max_k)

    def sat(self, kernel, vec_label_fn, atom_dict):
        """Delegate satisfaction evaluation to the desugared inner formula."""
        return self._inner.sat(kernel, vec_label_fn, atom_dict)

class Eventually(BoundedPCTLFormula):
    r"""Bounded PCTL :math:`F` (eventually) operator with a probability threshold.

    Semantics (informal):
        :math:`\mathbb{P}_{\geq p}[ F^{\leq B} \Phi ]` holds if :math:`\Phi` becomes true within :math:`B` steps with
        probability at least :math:`p`.

    This is implemented as a bounded until:
        :\math:`F^{\leq B} \Phi` is :math:`\top U^{\leq B} \Phi`.

    Args:
        prob: Probability threshold in ``[0, 1]``.
        bound: Local bound :math:`B`.
        subformula: Subformula :math:`\Phi`.

    Attributes:
        prob: Probability threshold.
        bound_param: Local bound :math:`B`.
        subformula: Subformula :math:`\Phi`.
    """

    def __init__(self, prob: float, bound: int, subformula: BoundedPCTLFormula):
        super().__init__()
        self.prob = float(prob)
        self.bound_param = int(bound)
        self.subformula = subformula

        self._inner = Until(
            prob=self.prob,
            bound=self.bound_param,
            subformula_1=Truth(),
            subformula_2=self.subformula,
        )

    @property
    def _bound(self) -> int:
        """Total bound delegated to the desugared inner formula."""
        return self._inner.bound

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        """Delegate sequence computation to the desugared inner formula."""
        return self._inner._prob_seq(kernel, vec_label_fn, atom_dict, max_k)

    def sat(self, kernel, vec_label_fn, atom_dict):
        """Delegate satisfaction evaluation to the desugared inner formula."""
        return self._inner.sat(kernel, vec_label_fn, atom_dict)

class Truth(BoundedPCTLFormula):
    """Boolean constant ``True`` formula."""

    @property
    def _bound(self):
        """Truth has zero temporal bound."""
        return 0

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Return 1.0 for all states.

        Args:
            kernel: Transition kernel used only to infer ``n_states``.
            vec_label_fn: Unused.
            atom_dict: Unused.

        Returns:
            Float64 array of ones with shape ``(n_states,)``.
        """
        n_states = kernel_n_states(kernel)
        return np.ones(n_states, dtype=np.float64)

class Atom(BoundedPCTLFormula):
    """Atomic proposition formula.

    Args:
        atom: Name of the atomic predicate.

    Attributes:
        atom: Predicate name, used to look up the row in ``vec_label_fn`` using
            ``atom_dict``.
    """

    def __init__(self, atom: str):
        super().__init__()
        self.atom = atom

    @property
    def _bound(self):
        """Atoms have zero temporal bound."""
        return 0

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Return the per-state truth values of this atom.

        Args:
            kernel: Unused (present for uniform signature).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, n_states)``.
            atom_dict: Mapping from atom name to ``vec_label_fn`` row index.

        Returns:
            Float64 array ``(n_states,)`` with values in ``{0.0, 1.0}``.
        """
        return vec_label_fn[atom_dict[self.atom]]

class Neg(BoundedPCTLFormula):
    r"""Logical negation formula :math:`\neq \Phi`.

    Args:
        subformula: The subformula :math:`\Phi` to negate.
    """

    def __init__(self, subformula: BoundedPCTLFormula):
        super().__init__()
        self.subformula = subformula

    @property
    def _bound(self):
        """Negation does not add temporal bound."""
        return self.subformula.bound

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        """Compute ``1 - sub_seq`` for the negated subformula."""
        seq = self.subformula._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return 1.0 - seq

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Evaluate ``1 - sat(subformula)`` per state."""
        return 1.0 - self.subformula.sat(kernel, vec_label_fn, atom_dict)

class And(BoundedPCTLFormula):
    r"""Logical conjunction formula :math:`\Phi_1 \land \Phi_2`.

    Args:
        subformula_1: Left conjunct :math:`\Phi_1`.
        subformula_2: Right conjunct :math:`\Phi_2`.

    Notes:
        This module uses multiplication as conjunction for ``{0,1}``-valued
        satisfaction arrays.
    """

    def __init__(self, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        """Conjunction bound is the max of operand bounds."""
        return max(self.subformula_1.bound, self.subformula_2.bound)

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        """Compute the elementwise product of operand sequences."""
        seq1 = self.subformula_1._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        seq2 = self.subformula_2._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return seq1 * seq2

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Evaluate the conjunction per state."""
        return (
            self.subformula_1.sat(kernel, vec_label_fn, atom_dict)
            * self.subformula_2.sat(kernel, vec_label_fn, atom_dict)
        )

class Or(BoundedPCTLFormula):
    r"""Logical disjunction formula :math:`\Phi_1 \lor \Phi_2`.

    Args:
        subformula_1: Left operand :math:`\Phi_1`.
        subformula_2: Right operand :math:`\Phi_2`.

    Notes:
        Satisfaction is computed via De Morgan's law or the probabilistic-style
        union formula on ``{0,1}`` sequences:
            ``1 - (1 - a) * (1 - b)``.
    """

    def __init__(self, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        """Disjunction bound is the max of operand bounds."""
        return max(self.subformula_1.bound, self.subformula_2.bound)

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Evaluate the disjunction per state."""
        return Neg(And(Neg(self.subformula_1), Neg(self.subformula_2))).sat(
            kernel, vec_label_fn, atom_dict
        )

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        """Compute disjunction sequence using ``1 - (1-a)*(1-b)``."""
        seq1 = self.subformula_1._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        seq2 = self.subformula_2._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return 1.0 - (1.0 - seq1) * (1.0 - seq2)

class BoundedPCTLModelChecker:
    """Shared base for bounded PCTL model checkers.

    This class stores:
      * A bounded PCTL formula.
      * A labeling function (state -> set of atoms) and a vectorized labeling
        matrix ``vec_label_fn``.
      * A transition kernel in one of two forms:

        1) **Full / dense MDP kernel** (``mode="full"``):
           ``transition_matrix`` with shape ``(n_states, n_states, n_actions)``.
           The first axis is next-state, the second is current-state, and the
           third is action (consistent with downstream einsum usage).

        2) **Compact successor representation** (``mode="compact"``):
           ``successor_states`` with shape ``(K, n_states)`` and ``probabilities``
           with shape ``(K, n_states, n_actions)``.

    Args:
        formula: The bounded PCTL formula to evaluate.
        label_fn: A ``LabelFn`` mapping ``state -> set[str]`` of atomic predicates.
        atomic_predicates: List of all atom names used by the label function and
            formulas.
        transition_matrix: Optional dense transition tensor
            ``(n_states, n_states, n_actions)``.
        successor_states: Optional successor state ids ``(K, n_states)`` (compact).
        probabilities: Optional successor probabilities ``(K, n_states, n_actions)``
            (compact).

    Attributes:
        formula: The stored formula.
        label_fn: Labeling function.
        atomic_predicates: List of atom names.
        atom_dict: Mapping ``atom -> index``.
        mode: ``"full"`` or ``"compact"``.
        n_states: Number of states.
        n_actions: Number of actions.
        vec_label_fn: Vectorized labeling matrix ``(n_atoms, n_states)``.

    Raises:
        AssertionError: If kernel shapes are inconsistent.
    """

    def __init__(
        self,
        formula: BoundedPCTLFormula,
        label_fn: LabelFn,
        atomic_predicates: List[str],
        transition_matrix: Optional[np.ndarray] = None,
        successor_states: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ):

        self.formula = formula
        self.label_fn = label_fn
        self.atomic_predicates = list(atomic_predicates)

        self.atom_dict = {atom: i for i, atom in enumerate(self.atomic_predicates)}

        if transition_matrix is not None:
            tm = np.asarray(transition_matrix, dtype=np.float64)
            assert tm.ndim == 3
            S, S2, A = tm.shape
            assert S == S2
            self.mode = "full"
            self.transition_matrix = tm
            self.successor_states = None
            self.probabilities = None
            self.n_states, self.n_actions = S, A
        else:
            assert successor_states is not None and probabilities is not None
            succ = np.asarray(successor_states, dtype=np.int64)
            probs = np.asarray(probabilities, dtype=np.float64)
            assert succ.ndim == 2
            assert probs.ndim == 3
            K, S = succ.shape
            K2, S2, A = probs.shape
            assert K == K2 and S == S2
            self.mode = "compact"
            self.transition_matrix = None
            self.successor_states = succ
            self.probabilities = probs
            self.n_states, self.n_actions = S, A

        self.vec_label_fn = self._build_vec_label_fn()

    def _build_vec_label_fn(self) -> np.ndarray:
        """Vectorize ``label_fn`` into a dense ``(n_atoms, n_states)`` array.

        Returns:
            A float64 array ``vec`` such that ``vec[i, s] == 1.0`` iff atom ``i``
            holds in state ``s``.
        """
        n_atoms = len(self.atomic_predicates)
        vec = np.zeros((n_atoms, self.n_states), dtype=np.float64)
        for s in range(self.n_states):
            labels = self.label_fn(s) 
            for atom, idx in self.atom_dict.items():
                vec[idx, s] = 1.0 if atom in labels else 0.0
        return vec

    def update_kernel(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        successor_states: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ):
        """Update the stored transition representation in-place.

        Exactly one of the following update modes should be used:

        - Dense update: provide ``transition_matrix`` with the same shape as the
          original dense matrix.
        - Compact update: provide both ``successor_states`` and ``probabilities``
          with shapes consistent with the original compact representation and
          the stored ``(n_states, n_actions)``.

        Args:
            transition_matrix: New dense kernel ``(n_states, n_states, n_actions)``.
            successor_states: New successor ids ``(K, n_states)`` (compact).
            probabilities: New successor probabilities ``(K, n_states, n_actions)``.

        Raises:
            AssertionError: If shapes are inconsistent with the originally
                configured representation.
        """
        if transition_matrix is not None:
            assert self.transition_matrix is not None
            tm = np.asarray(transition_matrix, dtype=np.float64)
            assert tm.ndim == 3
            S, S2, A = tm.shape
            assert S == S2
            assert tm.shape == self.transition_matrix.shape
            self.transition_matrix = tm
            self.successor_states = None
            self.probabilities = None
        else:
            assert self.successor_states is not None and self.probabilities is not None
            assert successor_states is not None and probabilities is not None
            succ = np.asarray(successor_states, dtype=np.int64)
            probs = np.asarray(probabilities, dtype=np.float64)
            assert succ.ndim == 2
            assert probs.ndim == 3
            K, S = succ.shape
            K2, S2, A = probs.shape
            assert K == K2 and S == S2
            self.transition_matrix = None
            assert S == self.n_states and A == self.n_actions
            self.successor_states = succ
            self.probabilities = probs

class ExactModelChecker(BoundedPCTLModelChecker):
    """Exact bounded PCTL model checker for a fixed policy.

    This checker:
      1. Collapses an MDP into a Markov chain by applying a stochastic policy.
      2. Evaluates the stored bounded PCTL formula on the resulting chain.

    Notes:
        - ``check_state(...)`` returns per-state satisfaction (vector over all
          states) for the policy-induced chain.
        - ``check_state_action(...)`` returns a state-action value-like quantity
          at horizon ``B-1`` (derived from ``_prob_seq``) corresponding to the
          expected satisfaction after taking an action then following the
          policy.
    """

    def __init__(
        self,
        formula: BoundedPCTLFormula,
        label_fn: LabelFn,
        atomic_predicates: List[str],
        transition_matrix: Optional[np.ndarray] = None,
        successor_states: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ):

        super().__init__(
            formula, 
            label_fn, 
            atomic_predicates, 
            transition_matrix=transition_matrix,
            successor_states=successor_states,
            probabilities=probabilities,
        )

    def check_state(
        self,
        key: jax.Array, 
        policy: np.array
    ) -> np.ndarray:
        """Evaluate the formula on the policy-induced Markov chain.

        Args:
            key: Unused random key (kept for API symmetry with SMC).
            policy: Stochastic policy as either:
                - shape ``(n_actions, n_states)`` (action-major), or
                - shape ``(n_states, n_actions)`` (state-major).
                Each row/column should sum to 1 for a proper policy.

        Returns:
            Float64 array ``(n_states,)`` in ``{0.0, 1.0}`` indicating which states
            satisfy the formula under the policy-induced chain.

        Raises:
            ValueError: If the policy shape is not recognized.
        """

        policy = np.asarray(policy, dtype=np.float64)

        tm = self.transition_matrix

        if policy.shape == (A, S):
            policy_sa = policy.T
            m_pi = np.einsum('ija,ai->ij', tm, policy)
        elif policy.shape == (S, A):
            policy_sa = policy
            m_pi = np.einsum('ija,ia->ij', tm, policy)
        else:
            raise ValueError(
                f"Unexpected policy shape {policy.shape}; expected "
                f"(n_actions, n_states) or (n_states, n_actions)"
            )

        if self.mode == "full":
            tm = self.transition_matrix
            m_pi = np.einsum("nsa,sa->ns", tm, policy_sa).astype(np.float64)
            kernel: Kernel = m_pi

        else:
            succ = self.successor_states
            probs = self.probabilities
            p_pi = np.einsum("ksa,sa->ks", probs, policy_sa).astype(np.float64)
            kernel = (succ, p_pi)
        

        return self.formula.sat(kernel, self.vec_label_fn, self.atom_dict)

    def check_state_action(
        self,
        key: jax.Array,
        policy: np.ndarray,
    ) -> np.ndarray:
        """Compute a state-action satisfaction value for the stored formula.

        This method computes the formula probability sequence up to the formula
        bound ``B``, then uses the ``(B-1)`` vector as a value function and performs
        one-step expectation under each action to produce a ``(n_states, n_actions)``
        array.

        Args:
            key: Unused random key (kept for API symmetry with SMC).
            policy: Stochastic policy as either ``(n_actions, n_states)`` or
                ``(n_states, n_actions)``.

        Returns:
            Float64 array ``Q`` with shape ``(n_states, n_actions)`` where ``Q[s, a]``
            is the expected satisfaction value after taking action ``a`` in state
            ``s`` and then following ``policy``.
        """

        policy = np.asarray(policy, dtype=np.float64)

        if policy.shape == (self.n_actions, self.n_states):
            policy_sa = policy.T
        elif policy.shape == (self.n_states, self.n_actions):
            policy_sa = policy
        else:
            raise ValueError(
                f"Unexpected policy shape {policy.shape}; expected "
                f"(n_actions, n_states) or (n_states, n_actions)"
            )

        if self.mode == "full":
            tm = self.transition_matrix
            m_pi = np.einsum('nsa,sa->ns', tm, policy_sa)
            kernel: Kernel = m_pi
        else:
            succ = self.successor_states
            probs = self.probabilities
            p_pi = np.einsum("ksa,sa->ks", probs, policy_sa).astype(np.float64)
            kernel = (succ, p_pi)
            
        B = self.formula.bound
        seq = self.formula._prob_seq(kernel, self.vec_label_fn, self.atom_dict, max_k=B)
        V_Bm1 = seq[max(B - 1, 0)]

        if self.mode == "full":
            tm = self.transition_matrix
            Q = np.einsum("nsa,n->sa", tm, V_Bm1).astype(np.float64)
        else:
            succ = self.successor_states
            probs = self.probabilities
            V_succ = V_Bm1[succ]
            Q = np.sum(probs * V_succ[:, :, None], axis=0).astype(np.float64)

        return Q

class StatisticalModelChecker(BoundedPCTLModelChecker):
    """Statistical model checker (SMC) for bounded PCTL formulas.

    This checker estimates satisfaction probabilities by Monte Carlo sampling of
    trajectories under a policy, then comparing the estimate ``p_hat`` to the
    formula's probability threshold.

    Supported formulas:
        - Probabilistic path operators: ``Next``, ``Until``, ``Eventually``, ``Always``.
        - Pure state formulas (``Truth``, ``Atom``, ``Neg``, ``And``, ``Or``, ``Implies``)
          are evaluated exactly without sampling.

    Notes:
        - Nested probabilistic operators inside state formulas are not supported
          (enforced in ``_eval_state_formula_python`` and ``_eval_state_formula_jax``).
        - Sampling is implemented in JAX for batching and speed.
    """

    def __init__(
        self,
        formula: BoundedPCTLFormula,
        label_fn: LabelFn,
        atomic_predicates: List[str],
        transition_matrix: Optional[np.ndarray] = None,
        successor_states: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ):
        super().__init__(
            formula, 
            label_fn, 
            atomic_predicates, 
            transition_matrix=transition_matrix,
            successor_states=successor_states,
            probabilities=probabilities,
        )

        self.vec_label_fn_jax = jnp.asarray(self.vec_label_fn, dtype=jnp.float64)

    def check_state(
        self,
        key: jax.Array, 
        policy: np.array,
        state: int,
        num_samples: int,
    ) -> np.ndarray:
        """Estimate whether a given state satisfies the formula under a policy.

        For probabilistic formulas, this:
          1) Samples ``num_samples`` trajectories up to ``max_steps``,
          2) Evaluates path satisfaction,
          3) Uses the sample mean as ``p_hat``,
          4) Returns ``1.0`` if ``p_hat >= prob_threshold`` else ``0.0``.

        For non-probabilistic formulas, the state formula is evaluated exactly.

        Args:
            key: PRNG key for sampling.
            policy: Stochastic policy as ``(n_actions, n_states)`` or
                ``(n_states, n_actions)``.
            state: Start state index.
            num_samples: Number of trajectories to sample.

        Returns:
            Scalar float64 array, either ``1.0`` (satisfies) or ``0.0`` (does not).
        """
        if not self._is_probabilistic_formula(self.formula):
            val = self._eval_state_formula_python(self.formula, state)
            return jnp.array(float(val), dtype=jnp.float64)

        start_state = int(state)
        num_samples = int(num_samples)
        max_steps = max(1, int(self.formula.bound))

        policy_sa = self._prepare_policy_probs_jax(
            self.n_states, self.n_actions, policy
        )

        prob_threshold = self._get_formula_prob(self.formula)

        if self.mode == "full":
            tm = self.transition_matrix
            m_pi = np.einsum("nsa,sa->ns", tm, policy_sa).astype(np.float64)
            m_pi_j = jnp.asarray(m_pi, dtype=jnp.float64)

            p_hat = self._estimate_prob_dense(
                key=key,
                start_state=start_state,
                num_samples=num_samples,
                max_steps=max_steps,
                m_first=m_pi_j,
                m_rest=m_pi_j,
                formula=self.formula,
                vec_labels=self.vec_label_fn_jax,
                atom_dict=self.atom_dict
            )
        else:
            assert self.successor_states is not None and self.probabilities is not None
            succ = self.successor_states
            probs = self.probabilities

            p_pi = np.einsum("ksa,sa->ks", probs, policy_sa).astype(np.float64)
            p_pi_j = jnp.asarray(p_pi, dtype=jnp.float64)

            p_hat = self._estimate_prob_compact(
                key=key,
                start_state=start_state,
                num_samples=num_samples,
                max_steps=max_steps,
                succ=succ_j,
                p_first=p_pi_j,
                p_rest=p_pi_j,
                formula=self.formula,
                vec_labels=self.vec_label_fn_jax,
                atom_dict=self.atom_dict
            )

        return jnp.where(p_hat >= prob_threshold, 1.0, 0.0).astype(jnp.float64)

    def check_state_action(
        self,
        key: jax.Array,
        policy: np.ndarray,
        state: int,
        action: int,
        num_samples: int,
    ) -> np.ndarray:
        """Estimate satisfaction when forcing the first action, then following a policy.

        For probabilistic formulas, the first transition uses action ``action``,
        and subsequent transitions follow ``policy``. For non-probabilistic state
        formulas, evaluation is exact and ignores ``action``/sampling.

        Args:
            key: PRNG key for sampling.
            policy: Stochastic policy as ``(n_actions, n_states)`` or
                ``(n_states, n_actions)``.
            state: Start state index.
            action: Action index to force at the first step.
            num_samples: Number of trajectories to sample.

        Returns:
            Scalar float64 array, either ``1.0`` (satisfies) or ``0.0`` (does not).
        """
        if not self._is_probabilistic_formula(self.formula):
            val = self._eval_state_formula_python(self.formula, state)
            return jnp.array(float(val), dtype=jnp.float64)

        start_state = int(state)
        forced_action = int(action)
        num_samples = int(num_samples)
        max_steps = max(1, int(self.formula.bound))

        policy_sa = self._prepare_policy_probs_jax(
            self.n_states, self.n_actions, policy
        )

        prob_threshold = self._get_formula_prob(self.formula)

        if self.mode == "full":
            tm = self.transition_matrix
            m_a = tm[:, :, action]
            m_pi = np.einsum("nsa,sa->ns", tm, policy_sa).astype(np.float64)
            
            m_a_j = jnp.asarray(m_a, dtype=jnp.float64)
            m_pi_j = jnp.asarray(m_pi, dtype=jnp.float64)

            p_hat = self._estimate_prob_dense(
                key=key,
                start_state=start_state,
                num_samples=num_samples,
                max_steps=max_steps,
                m_first=m_a_j,
                m_rest=m_pi_j,
                formula=self.formula,
                vec_labels=self.vec_label_fn_jax,
                atom_dict=self.atom_dict
            )
        else:
            assert self.successor_states is not None and self.probabilities is not None
            succ = self.successor_states
            probs = self.probabilities

            p_a = probs[:, :, action]
            p_pi = np.einsum("ksa,sa->ks", probs, policy_sa).astype(np.float64)

            succ_j = jnp.asarray(succ, dtype=jnp.int64)
            p_a_j = jnp.asarray(p_a, dtype=jnp.float64)
            p_pi_j = jnp.asarray(p_pi, dtype=jnp.float64)

            p_hat = self._estimate_prob_compact(
                key=key,
                start_state=start_state,
                num_samples=num_samples,
                max_steps=max_steps,
                succ=succ_j,
                p_first=p_a_j,
                p_rest=p_pi_j,
                formula=self.formula,
                vec_labels=self.vec_label_fn_jax,
                atom_dict=self.atom_dict
            )

        return jnp.where(p_hat >= prob_threshold, 1.0, 0.0).astype(jnp.float64)

    def _is_probabilistic_formula(self, formula: BoundedPCTLFormula) -> bool:
        """Return True if ``formula`` is a probabilistic path operator."""
        return isinstance(formula, (Next, Until, Eventually, Always))

    def _get_formula_prob(self, formula: BoundedPCTLFormula) -> float:
        """Extract probability threshold from a probabilistic formula.

        Args:
            formula: A probabilistic formula (``Next``, ``Until``, ``Eventually``, ``Always``).

        Returns:
            Probability threshold.

        Raises:
            ValueError: If ``formula`` is not a probabilistic operator.
        """
        if isinstance(formula, (Next, Until, Eventually, Always)):
            return float(formula.prob)
        raise ValueError(
            "Trying to get probability threshold from non-probabilistic formula."
        )

    def _eval_state_formula_python(self, formula: BoundedPCTLFormula, state: int) -> bool:
        """Evaluate a pure state formula in Python.

        This supports only boolean (non-probabilistic) formulas. Nested
        probabilistic operators are explicitly rejected.

        Args:
            formula: Formula to evaluate.
            state: State index.

        Returns:
            Boolean satisfaction.

        Raises:
            NotImplementedError: If nested probabilistic operators are encountered.
            TypeError: If an unsupported formula type is encountered.
        """
        if isinstance(formula, Truth):
            return True
        if isinstance(formula, Atom):
            idx = self.atom_dict[formula.atom]
            return bool(self.vec_label_fn[idx, state] > 0.5)
        if isinstance(formula, Neg):
            return not self._eval_state_formula_python(formula.subformula, state)
        if isinstance(formula, And):
            return (
                self._eval_state_formula_python(formula.subformula_1, state)
                and self._eval_state_formula_python(formula.subformula_2, state)
            )
        if isinstance(formula, Or):
            return (
                self._eval_state_formula_python(formula.subformula_1, state)
                or self._eval_state_formula_python(formula.subformula_2, state)
            )
        if isinstance(formula, Implies):
            left = self._eval_state_formula_python(formula.subformula_1, state)
            right = self._eval_state_formula_python(formula.subformula_2, state)
            return (not left) or right
        if isinstance(formula, (Next, Until, Eventually, Always)):
            raise NotImplementedError(
                "Nested probabilistic operators in state formulas are not "
                "supported in StatisticalModelChecker."
            )
        raise TypeError(f"Unsupported formula type in Python state eval: {type(formula)}")

    @staticmethod
    @partial(jit, static_argnames=["n_states", "n_actions"])
    def _prepare_policy_probs_jax(
        n_states: int, n_actions: int, policy: jnp.ndarray
    ) -> jnp.ndarray:
        """Normalize policy shape for JAX computation.

        Args:
            n_states: Number of states.
            n_actions: Number of actions.
            policy: Policy probabilities as either ``(n_actions, n_states)`` or
                ``(n_states, n_actions)``.

        Returns:
            Policy as a ``(n_states, n_actions)`` JAX array.

        Raises:
            ValueError: If ``policy`` has an incompatible shape.
        """
        policy = jnp.asarray(policy, dtype=jnp.float64)
        if policy.shape == (n_actions, n_states):
            policy = policy.T
        elif policy.shape == (n_states, n_actions):
            pass
        else:
            raise ValueError(
                f"Policy shape {policy.shape} incompatible with "
                f"(n_states, n_actions)=({n_states}, {n_actions})"
            )
        return policy

    @staticmethod
    def _eval_state_formula_jax(
        state_idx: jnp.ndarray, 
        formula: BoundedPCTLFormula,
        vec_labels: jnp.ndarray, 
        atom_dict: Dict[str, int], 
    ) -> jnp.ndarray:
        """Evaluate a pure state formula in JAX.

        Args:
            state_idx: Scalar state index.
            formula: State formula (must not contain probabilistic operators).
            vec_labels: Vectorized labels ``(n_atoms, n_states)`` as a JAX array.
            atom_dict: Mapping from atom name to row index in ``vec_labels``.

        Returns:
            JAX boolean scalar indicating satisfaction.

        Raises:
            NotImplementedError: If nested probabilistic operators are encountered.
            TypeError: If an unsupported formula type is encountered.
        """

        def rec(f: BoundedPCTLFormula) -> jnp.ndarray:
            if isinstance(f, Truth):
                return jnp.bool_(True)
            if isinstance(f, Atom):
                idx = atom_dict[f.atom]
                return vec_labels[idx, state_idx] > 0.5
            if isinstance(f, Neg):
                return jnp.logical_not(rec(f.subformula))
            if isinstance(f, And):
                return jnp.logical_and(
                    rec(f.subformula_1),
                    rec(f.subformula_2),
                )
            if isinstance(f, Or):
                return jnp.logical_or(
                    rec(f.subformula_1),
                    rec(f.subformula_2),
                )
            if isinstance(f, Implies):
                left = rec(f.subformula_1)
                right = rec(f.subformula_2)
                return jnp.logical_or(jnp.logical_not(left), right)
            if isinstance(f, (Next, Until, Eventually, Always)):
                raise NotImplementedError(
                    "Nested probabilistic operators in state formulas "
                    "are not supported in JAX SMC."
                )
            raise TypeError(f"Unsupported formula type in JAX state eval: {type(f)}")

        return rec(formula)

    @staticmethod
    def _path_satisfies_single(
        states_1d: jnp.ndarray,
        formula: BoundedPCTLFormula,
        vec_labels: jnp.ndarray, 
        atom_dict: Dict[str, int], 
    ) -> jnp.ndarray:
        """Evaluate whether a single sampled path satisfies the formula.

        Args:
            states_1d: A single trajectory of state indices with shape ``(T+1,)``.
            formula: The (bounded) formula to check.
            vec_labels: Vectorized labels ``(n_atoms, n_states)`` as a JAX array.
            atom_dict: Mapping from atom name to row index.

        Returns:
            JAX boolean scalar: True iff the trajectory satisfies ``formula``.
        """
        eval_state = StatisticalModelChecker._eval_state_formula_jax

        if isinstance(formula, Next):
            s1 = states_1d[1]
            return eval_state(
                s1, formula.subformula, vec_labels, atom_dict
            )

        if isinstance(formula, Until):
            B = formula.bound
            idxs = jnp.arange(B + 1)
            s_seq = states_1d[idxs]
            sat1 = jax.vmap(lambda s: eval_state(
                s, formula.subformula_1, vec_labels, atom_dict
            ))(s_seq)
            sat2 = jax.vmap(lambda s: eval_state(
                s, formula.subformula_2, vec_labels, atom_dict
            ))(s_seq)

            def scan_body(carry, x):
                new_carry = jnp.logical_and(carry, x)
                return new_carry, new_carry

            init = jnp.bool_(True)
            _, prefix_tail = jax.lax.scan(scan_body, init, sat1[:-1])
            prefix_all = jnp.concatenate(
                [jnp.array([True], dtype=bool), prefix_tail], axis=0
            )
            cond_k = jnp.logical_and(sat2, prefix_all)
            return jnp.any(cond_k)

        if isinstance(formula, Eventually):
            B = formula.bound_param
            idxs = jnp.arange(B + 1)
            s_seq = states_1d[idxs]
            sat = jax.vmap(lambda s: eval_state(
                s, formula.subformula, vec_labels, atom_dict
            ))(s_seq)
            return jnp.any(sat)

        if isinstance(formula, Always):
            B = formula.bound_param
            idxs = jnp.arange(B + 1)
            s_seq = states_1d[idxs]
            sat = jax.vmap(lambda s: eval_state(
                s, formula.subformula, vec_labels, atom_dict
            ))(s_seq)
            return jnp.all(sat)

        return eval_state(states_1d[0], formula, vec_labels, atom_dict)

    @staticmethod
    def _estimate_prob_dense(
        key: jax.Array,
        start_state: int,
        num_samples: int,
        max_steps: int,
        m_first: jnp.ndarray,
        m_rest: jnp.ndarray,
        formula: BoundedPCTLFormula,
        vec_labels: jnp.ndarray,
        atom_dict: Dict[str, int],
    ) -> jnp.ndarray:
        """Estimate satisfaction probability by sampling from a dense kernel.

        Args:
            key: PRNG key.
            start_state: Initial state index.
            num_samples: Number of trajectories to sample.
            max_steps: Trajectory horizon (number of transitions).
            m_first: Transition matrix for the first step ``(n_states, n_states)``.
            m_rest: Transition matrix for subsequent steps ``(n_states, n_states)``.
            formula: Formula to check along sampled paths.
            vec_labels: Vectorized labels ``(n_atoms, n_states)`` as a JAX array.
            atom_dict: Atom-to-index mapping.

        Returns:
            Scalar float64 JAX array: estimated probability ``p_hat``.
        """
        states_batch = StatisticalModelChecker._sample_trajectories_dense_jax(
            key=key,
            m_first=m_first,
            m_rest=m_rest,
            start_state=jnp.asarray(start_state, dtype=jnp.float64),
            max_steps=max_steps,
            num_samples=num_samples,
        )

        path_satisfies_batch = jax.vmap(
            lambda s: StatisticalModelChecker._path_satisfies_single(
                s, formula, vec_labels, atom_dict
            ),
            in_axes=0,
        )

        sat_batch = path_satisfies_batch(states_batch)  # (num_samples,)
        return jnp.mean(sat_batch.astype(jnp.float64))

    @staticmethod
    def _estimate_prob_compact(
        key: jax.Array,
        start_state: int,
        num_samples: int,
        max_steps: int,
        succ: jnp.ndarray,
        p_first: jnp.ndarray,
        p_rest: jnp.ndarray,
        formula: BoundedPCTLFormula,
        vec_labels: jnp.ndarray,
        atom_dict: Dict[str, int],
    ) -> jnp.ndarray:
        """Estimate satisfaction probability by sampling from a compact kernel.

        Args:
            key: PRNG key.
            start_state: Initial state index.
            num_samples: Number of trajectories to sample.
            max_steps: Trajectory horizon (number of transitions).
            succ: Successor matrix ``(K, n_states)``.
            p_first: Probabilities for the first step ``(K, n_states)``.
            p_rest: Probabilities for subsequent steps ``(K, n_states)``.
            formula: Formula to check along sampled paths.
            vec_labels: Vectorized labels ``(n_atoms, n_states)`` as a JAX array.
            atom_dict: Atom-to-index mapping.

        Returns:
            Scalar float64 JAX array: estimated probability ``p_hat``.
        """
        states_batch = StatisticalModelChecker._sample_trajectories_compact_jax(
            key=key,
            succ=succ,
            p_first=p_first,
            p_rest=p_rest,
            start_state=jnp.asarray(start_state, dtype=jnp.float64),
            max_steps=max_steps,
            num_samples=num_samples,
        )

        path_satisfies_batch = jax.vmap(
            lambda s: StatisticalModelChecker._path_satisfies_single(
                s, formula, vec_labels, atom_dict
            ),
            in_axes=0,
        )

        sat_batch = path_satisfies_batch(states_batch)  # (num_samples,)
        return jnp.mean(sat_batch.astype(jnp.float64))

    @staticmethod
    @partial(jit, static_argnames=["num_samples", "max_steps"])
    def _sample_trajectories_dense_jax(
        key: jax.Array,
        m_first: jnp.ndarray,
        m_rest: jnp.ndarray,
        start_state: jnp.ndarray,
        max_steps: int,
        num_samples: int,
    ) -> jnp.ndarray:
        """Sample a batch of trajectories from a dense transition kernel.

        The returned tensor is shaped ``(num_samples, max_steps + 1)``.

        Special handling:
            - Rows with zero outgoing probability mass fall back to a self-loop
              (the agent stays in the same state).

        Args:
            key: PRNG key.
            m_first: Transition matrix for the first step ``(n_states, n_states)``.
            m_rest: Transition matrix for subsequent steps ``(n_states, n_states)``.
            start_state: Scalar start state (integer-valued, stored as array).
            max_steps: Number of transitions to sample.
            num_samples: Number of independent trajectories.

        Returns:
            Integer JAX array of shape ``(num_samples, max_steps + 1)``.
        """
        S, S2 = m_rest.shape
        assert S == S2
        assert (S, S2) == m_first.shape

        states0 = jnp.full((num_samples,), start_state, dtype=jnp.int64)

        def body(carry, t):
            states, key = carry
            key, subkey = jax.random.split(key)

            kernel = jax.lax.cond(
                t == 0,
                lambda _: m_first,
                lambda _: m_rest,
                operand=None,
            )

            probs = kernel[:, states].T  # (N, S)

            row_sums = probs.sum(axis=-1, keepdims=True)  # (N,1)
            has_mass = row_sums > 0.0

            fallback = jax.nn.one_hot(states, S)  # (N, S)

            denom = jnp.where(row_sums > 0.0, row_sums, 1.0)

            norm_probs = jnp.where(
                has_mass,
                probs / denom,   # valid rows
                fallback,        # zero-mass rows
            )

            logits = jnp.where(norm_probs > 0, jnp.log(norm_probs), -jnp.inf)

            next_states = jax.random.categorical(subkey, logits=logits, axis=-1)
            next_states = next_states.astype(jnp.int64)

            return (next_states, key), next_states
            
        
        ts = jnp.arange(max_steps, dtype=jnp.int64)
        (final_states, _), states_seq = jax.lax.scan(body, (states0, key), ts)
        states_all = jnp.concatenate([states0[None, :], states_seq], axis=0)
        return states_all.T # expected output shape (N, T+1)

    @staticmethod
    @partial(jit, static_argnames=["num_samples", "max_steps"])
    def _sample_trajectories_compact_jax(
        key: jax.Array,
        succ: jnp.ndarray,
        p_first: jnp.ndarray,
        p_rest: jnp.ndarray,
        start_state: jnp.ndarray,
        max_steps: int,
        num_samples: int,
    ) -> jnp.ndarray:
        """Sample a batch of trajectories from a compact successor kernel.

        The returned tensor is shaped ``(num_samples, max_steps + 1)``.

        Special handling:
            - Rows with zero outgoing probability mass fall back to a self-loop
              (the agent stays in the same state).

        Args:
            key: PRNG key.
            succ: Successor matrix ``(K, n_states)``.
            p_first: Probabilities for first step ``(K, n_states)``.
            p_rest: Probabilities for subsequent steps ``(K, n_states)``.
            start_state: Scalar start state (integer-valued, stored as array).
            max_steps: Number of transitions to sample.
            num_samples: Number of independent trajectories.

        Returns:
            Integer JAX array of shape ``(num_samples, max_steps + 1)``.
        """
        states0 = jnp.full((num_samples,), start_state, dtype=jnp.int64)

        def body(carry, t):
            states, key = carry
            key, subkey = jax.random.split(key)

            p = jax.lax.cond(t == 0, lambda _: p_first, lambda _: p_rest, operand=None)
            probs = p[:, states].T

            # normalize rows (keep zeros as zeros; fallback self-loop if all-zero)
            row_sums = probs.sum(axis=-1, keepdims=True)
            has_mass = row_sums > 0
            denom = jnp.where(has_mass, row_sums, 1.0)
            norm = probs / denom

            # choose successor index k, then map to next state id
            logits = jnp.where(norm > 0, jnp.log(norm), -jnp.inf)
            k_idx = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int64)
            next_states = succ[k_idx, states]

            # if no mass, stay in place
            next_states = jnp.where(has_mass[:, 0], next_states, states)

            return (next_states, key), next_states

        ts = jnp.arange(max_steps, dtype=jnp.int64)
        (_, _), states_seq = jax.lax.scan(body, (states0, key), ts)
        states_all = jnp.concatenate([states0[None, :], states_seq], axis=0)
        return states_all.T # expected output shape (N, T+1)
