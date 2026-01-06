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
    r"""Return the number of states induced by a transition kernel.

    MASA evaluates bounded PCTL formulas on a Markov-chain transition kernel that
    can be represented in either a dense or compact form.

    **Dense kernel**
      A 2D transition matrix ``m`` with shape ``(n_states, n_states)`` where
      column ``s`` encodes the distribution over next states from state ``s``.
      (So ``m[:, s]`` is a categorical distribution when state ``s`` has outgoing
      probability mass.)

    **Compact kernel**
      A tuple ``(succ, p)`` where:

      - ``succ`` has shape ``(K, n_states)`` and stores up to ``K`` successor
        state ids per state.
      - ``p`` has shape ``(K, n_states)`` and stores aligned successor
        probabilities.

    Args:
        kernel: The transition kernel, either a dense matrix or a ``(succ, p)``
            tuple.

    Returns:
        The number of states ``n_states``.
    """
    if isinstance(kernel, tuple):
        succ, p = kernel
        return succ.shape[1]
    return kernel.shape[1]

class BoundedPCTLFormula:
    r"""Base class for bounded PCTL-style formulas.

    MASA represents bounded PCTL formulas as a tree of :class:`BoundedPCTLFormula`
    objects. A formula is evaluated on a *Markov chain* transition kernel (dense
    or compact) and a vectorized labeling matrix.

    The formula API centers around two related operations:

    - :meth:`sat`: evaluate the formula at its final time bound and return a
      per-state satisfaction indicator (a float array in ``{0.0, 1.0}``).
    - :meth:`_prob_seq`: compute a time-indexed sequence
      :math:`P_0(s), \dots, P_K(s)` up to a specified horizon.

    For propositional (state) formulas, the default :meth:`_prob_seq` repeats the
    time-0 satisfaction vector across all time steps.

    **Vectorized labels**
      A :class:`~masa.common.label_fn.LabelFn` maps each state to a set of atomic
      predicates (strings). The model checker precomputes:

      .. math::

         \mathrm{vec\_label\_fn} \in \{0,1\}^{|AP| \times |S|},

      where ``vec_label_fn[i, s] = 1`` iff atomic predicate ``AP[i]`` holds in
      state ``s``.

    Attributes:
        bound: Total time bound of the formula (including any nested subformula
            contributions).

    Notes:
        Implementations of temporal operators typically use JAX (e.g. ``jit``,
        ``vmap``, ``lax.scan``) to accelerate probability-sequence computation.
    """
    def __init__(self) -> None:
        pass

    @property
    def _bound(self) -> int:
        """Internal bound implementation.

        Subclasses must implement this property.

        Returns:
            The non-negative time bound.

        Raises:
            NotImplementedError: If a subclass does not implement this property.
        """
        raise NotImplementedError

    @property
    def bound(self) -> int:
        """Total time bound of this formula (read-only)."""
        return self._bound

    def _prob_seq(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
        max_k: int | None = None,
    ) -> np.ndarray:
        r"""Compute a satisfaction/probability sequence up to horizon ``max_k``.

        The returned sequence has shape ``(max_k + 1, n_states)`` where row ``k``
        corresponds to the value at time ``k``.

        For state formulas, the default implementation repeats :meth:`sat` across
        time, i.e. :math:`P_k(s) = P_0(s)`.

        Args:
            kernel: Markov-chain kernel (dense ``(S,S)`` or compact ``(succ,p)``).
            vec_label_fn: Vectorized labeling matrix of shape
                ``(n_atoms, n_states)``.
            atom_dict: Mapping from atomic predicate string to row index in
                ``vec_label_fn``.
            max_k: Sequence horizon (inclusive). If ``None``, defaults to
                :attr:`bound`.

        Returns:
            A float64 array of shape ``(max_k + 1, n_states)``.

        Notes:
            Temporal/probabilistic operators override this method.
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
        r"""Evaluate the formula at its bound for all states.

        Args:
            kernel: Markov-chain kernel (dense or compact).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, n_states)``.
            atom_dict: Mapping from atomic predicate string to row index.

        Returns:
            Float64 array of shape ``(n_states,)`` with values in ``{0.0, 1.0}``.
        """
        return self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound)[
            self.bound
        ]

class Next(BoundedPCTLFormula):
    r"""Bounded PCTL *next* operator :math:`X` with probability threshold.

    This represents:

    .. math::

       \mathbb{P}_{\ge p}[X\,\Phi].

    Informally, the formula holds at state :math:`s` iff the probability that
    :math:`\Phi` holds in the next state is at least :math:`p`.

    Args:
        prob: Probability threshold :math:`p \in [0,1]`.
        subformula: The subformula :math:`\Phi` evaluated at the next state.

    Attributes:
        prob: Probability threshold :math:`p`.
        subformula: Nested formula :math:`\Phi`.
        bound_param: Local bound contributed by this operator (fixed to 1).

    See Also:
        :meth:`_prob_seq`: Computes the shifted probability sequence for
        :math:`X\,\Phi`.
    """

    def __init__(self, prob: float, subformula: BoundedPCTLFormula):
        super().__init__()
        self.prob = prob
        self.subformula = subformula
        self.bound_param = 1

    @property
    def _bound(self):
        r"""Total bound for :class:`Next`.

        Returns:
            ``1 + subformula.bound``.
        """
        return self.bound_param + self.subformula.bound

    @staticmethod
    @jit
    def _next_prob_seq_core_dense(m: jnp.ndarray, sub_seq: jnp.ndarray) -> jnp.ndarray:
        r"""JAX core for :class:`Next` on a dense kernel.

        Args:
            m: Dense Markov-chain transition matrix of shape ``(S, S)``, where
                column ``s`` is the distribution over next states from state ``s``.
            sub_seq: Subformula sequence of shape ``(T, S)``.

        Returns:
            Sequence of shape ``(T + 1, S)``. Row 0 is all zeros (the shifted
            operator is defined to have probability 0 at time 0), and rows 1..T
            contain one-step expectations of ``sub_seq``.
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
        r"""JAX core for :class:`Next` on a compact kernel.

        Args:
            succ: Successor ids of shape ``(K, S)``.
            p: Successor probabilities of shape ``(K, S)`` aligned with ``succ``.
            sub_seq: Subformula sequence of shape ``(T, S)``.
            K: Max successors per state (static argument for JIT).

        Returns:
            Sequence of shape ``(T + 1, S)`` with row 0 all zeros.
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
        r"""Compute the probability sequence for :math:`X\,\Phi`.

        Args:
            kernel: Markov-chain kernel (dense or compact).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, S)``.
            atom_dict: Mapping from atom string to row index.
            max_k: Local horizon (inclusive). If ``None``, defaults to 1.

        Returns:
            Float64 array of shape ``(max_k + 1, S)``.

        Notes:
            If ``max_k == 0``, returns a single row of zeros.
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
        r"""Threshold the one-step probability for :math:`\mathbb{P}_{\ge p}[X\,\Phi]`.

        Args:
            kernel: Markov-chain kernel (dense or compact).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, S)``.
            atom_dict: Mapping from atom string to row index.

        Returns:
            Float64 array of shape ``(S,)`` in ``{0.0, 1.0}``.
        """
        probs = self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound_param)[
            self.bound_param
        ]
        return (probs >= self.prob).astype(np.float64)

class Until(BoundedPCTLFormula):
    r"""Bounded PCTL *until* operator :math:`U^{\le B}` with probability threshold.

    This represents:

    .. math::

       \mathbb{P}_{\ge p}[\Phi_1\ U^{\le B}\ \Phi_2].

    Informally, the formula holds at state :math:`s` iff the probability that
    :math:`\Phi_2` becomes true within :math:`B` steps while :math:`\Phi_1` holds
    at all preceding steps is at least :math:`p`.

    The bounded-until recurrence computed by :meth:`_prob_seq` is:

    .. math::

       P_0 &= \mathrm{sat}_2, \\\\
       P_{k+1} &= \mathrm{sat}_2 + \bigl((1-\mathrm{sat}_2)\,\mathrm{sat}_1\bigr)\,
                 \mathbb{E}[P_k(s')].

    Args:
        prob: Probability threshold :math:`p \in [0,1]`.
        bound: Local bound :math:`B`.
        subformula_1: Continuation condition :math:`\Phi_1`.
        subformula_2: Target condition :math:`\Phi_2`.

    Attributes:
        prob: Probability threshold.
        bound_param: Local bound :math:`B`.
        subformula_1: Continuation formula.
        subformula_2: Target formula.
    """

    def __init__(self, prob: float, bound, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.prob = prob
        self.bound_param = int(bound)
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        r"""Total bound for :class:`Until`.

        Returns:
            ``bound_param + subformula_1.bound + subformula_2.bound``.
        """
        return self.bound_param + self.subformula_1.bound + self.subformula_2.bound

    @staticmethod
    @partial(jit, static_argnames=("max_k",))
    def _until_prob_seq_core_dense(
        m: jnp.ndarray, sat1: jnp.ndarray, sat2: jnp.ndarray, max_k: int
    ) -> jnp.ndarray:
        r"""JAX core for bounded-until probabilities using a dense kernel.

        Args:
            m: Dense Markov-chain transition matrix of shape ``(S, S)``.
            sat1: Satisfaction mask for :math:`\Phi_1`, shape ``(S,)`` in ``{0,1}``.
            sat2: Satisfaction mask for :math:`\Phi_2`, shape ``(S,)`` in ``{0,1}``.
            max_k: Local bound :math:`B`.

        Returns:
            Float64 JAX array of shape ``(max_k + 1, S)``.
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
        r"""JAX core for bounded-until probabilities using a compact kernel.

        Args:
            succ: Successor ids ``(K, S)``.
            p: Successor probabilities ``(K, S)``.
            sat1: Satisfaction mask for :math:`\Phi_1`, shape ``(S,)`` in ``{0,1}``.
            sat2: Satisfaction mask for :math:`\Phi_2`, shape ``(S,)`` in ``{0,1}``.
            max_k: Local bound :math:`B`.
            K: Max successors per state (static argument for JIT).

        Returns:
            Float64 JAX array of shape ``(max_k + 1, S)``.
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
        r"""Compute the bounded-until probability sequence.

        Args:
            kernel: Markov-chain kernel (dense or compact).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, S)``.
            atom_dict: Mapping from atom string to row index.
            max_k: Local horizon (inclusive). If ``None``, defaults to
                :attr:`bound_param`.

        Returns:
            Float64 array of shape ``(max_k + 1, S)``.
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
        r"""Threshold :math:`\mathbb{P}_{\ge p}[\Phi_1\ U^{\le B}\ \Phi_2]`.

        Args:
            kernel: Markov-chain kernel (dense or compact).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, S)``.
            atom_dict: Mapping from atom string to row index.

        Returns:
            Float64 array of shape ``(S,)`` in ``{0.0, 1.0}``.
        """
        probs = self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound_param)[
            self.bound_param
        ]
        return (probs >= self.prob).astype(np.float64)

class Always(BoundedPCTLFormula):
    r"""Bounded PCTL *always* operator :math:`G^{\le B}` with probability threshold.

    This represents:

    .. math::

       \mathbb{P}_{\ge p}[G^{\le B}\,\Phi].

    MASA implements bounded *always* via duality:

    .. math::

       G^{\le B}\Phi \equiv \neg(\top\ U^{\le B}\ \neg\Phi),

    with the threshold transformation :math:`p \mapsto 1-p` applied to the inner
    until.

    Args:
        prob: Probability threshold :math:`p \in [0,1]`.
        bound: Local bound :math:`B`.
        subformula: Subformula :math:`\Phi`.

    Attributes:
        prob: Probability threshold.
        bound_param: Local bound.
        subformula: Nested formula.
        _inner: Desugared formula (internal) built from :class:`Neg`,
            :class:`Until`, and :class:`Truth`.
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
        """Delegate probability-sequence computation to :attr:`_inner`."""
        return self._inner._prob_seq(kernel, vec_label_fn, atom_dict, max_k)

    def sat(self, kernel, vec_label_fn, atom_dict):
        """Delegate probability-sequence computation to :attr:`_inner`."""
        return self._inner.sat(kernel, vec_label_fn, atom_dict)

class Eventually(BoundedPCTLFormula):
    r"""Bounded PCTL *eventually* operator :math:`F^{\le B}` with probability threshold.

    This represents:

    .. math::

       \mathbb{P}_{\ge p}[F^{\le B}\,\Phi].

    MASA implements bounded *eventually* as a bounded until:

    .. math::

       F^{\le B}\Phi \equiv \top\ U^{\le B}\ \Phi.

    Args:
        prob: Probability threshold :math:`p \in [0,1]`.
        bound: Local bound :math:`B`.
        subformula: Subformula :math:`\Phi`.

    Attributes:
        prob: Probability threshold.
        bound_param: Local bound.
        subformula: Nested formula.
        _inner: Desugared formula (internal) built from :class:`Until` and
            :class:`Truth`.
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
        """Delegate probability-sequence computation to :attr:`_inner`."""
        return self._inner._prob_seq(kernel, vec_label_fn, atom_dict, max_k)

    def sat(self, kernel, vec_label_fn, atom_dict):
        """Delegate satisfaction evaluation to :attr:`_inner`."""
        return self._inner.sat(kernel, vec_label_fn, atom_dict)

class Truth(BoundedPCTLFormula):
    r"""Boolean constant :math:`\top` (true)."""

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
        """Return ``1.0`` for all states.

        Args:
            kernel: Transition kernel (used only to infer ``n_states``).
            vec_label_fn: Unused.
            atom_dict: Unused.

        Returns:
            Float64 array of ones with shape ``(n_states,)``.
        """
        n_states = kernel_n_states(kernel)
        return np.ones(n_states, dtype=np.float64)

class Atom(BoundedPCTLFormula):
    r"""Atomic proposition.

    The atom name is resolved via ``atom_dict`` and selects the corresponding row
    in ``vec_label_fn``.

    Args:
        atom: Name of the atomic predicate.

    Attributes:
        atom: Atomic predicate name.
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
        """Return the per-state truth values of this atomic predicate.

        Args:
            kernel: Unused (present for signature consistency).
            vec_label_fn: Vectorized labeling matrix ``(n_atoms, n_states)``.
            atom_dict: Mapping from atom string to row index.

        Returns:
            Float64 array of shape ``(n_states,)`` in ``{0.0, 1.0}``.
        """
        return vec_label_fn[atom_dict[self.atom]]

class Neg(BoundedPCTLFormula):
    r"""Logical negation :math:`\neg \Phi`.

    Args:
        subformula: Subformula :math:`\Phi`.

    Attributes:
        subformula: Nested formula :math:`\Phi`.
    """

    def __init__(self, subformula: BoundedPCTLFormula):
        super().__init__()
        self.subformula = subformula

    @property
    def _bound(self):
        """Negation does not add temporal bound."""
        return self.subformula.bound

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        """Compute ``1 -`` the subformula probability sequence.

        Args:
            kernel: Markov-chain kernel.
            vec_label_fn: Vectorized labeling matrix.
            atom_dict: Atom-to-row mapping.
            max_k: Sequence horizon.

        Returns:
            Float64 array ``1.0 - sub_seq`` with shape ``(max_k + 1, n_states)``.
        """
        seq = self.subformula._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return 1.0 - seq

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Compute ``1 -`` the subformula satisfaction indicator.

        Args:
            kernel: Markov-chain kernel.
            vec_label_fn: Vectorized labeling matrix.
            atom_dict: Atom-to-row mapping.

        Returns:
            Float64 array of shape ``(n_states,)`` in ``{0.0, 1.0}``.
        """
        return 1.0 - self.subformula.sat(kernel, vec_label_fn, atom_dict)

class And(BoundedPCTLFormula):
    r"""Logical conjunction :math:`\Phi_1 \land \Phi_2`.

    MASA uses multiplication as conjunction for ``{0,1}``-valued satisfaction
    arrays.

    Args:
        subformula_1: Left operand :math:`\Phi_1`.
        subformula_2: Right operand :math:`\Phi_2`.

    Attributes:
        subformula_1: Left operand.
        subformula_2: Right operand.
    """

    def __init__(self, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        """Bound is ``max(bound(subformula_1), bound(subformula_2))``."""
        return max(self.subformula_1.bound, self.subformula_2.bound)

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        """Compute the elementwise product of operand sequences.

        Args:
            kernel: Markov-chain kernel.
            vec_label_fn: Vectorized labeling matrix.
            atom_dict: Atom-to-row mapping.
            max_k: Sequence horizon.

        Returns:
            Float64 array of shape ``(max_k + 1, n_states)``.
        """
        seq1 = self.subformula_1._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        seq2 = self.subformula_2._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return seq1 * seq2

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Compute the conjunction per state.

        Returns:
            Float64 array of shape ``(n_states,)`` in ``{0.0, 1.0}``.
        """
        return (
            self.subformula_1.sat(kernel, vec_label_fn, atom_dict)
            * self.subformula_2.sat(kernel, vec_label_fn, atom_dict)
        )

class Or(BoundedPCTLFormula):
    r"""Logical disjunction :math:`\Phi_1 \lor \Phi_2`.

    For ``{0,1}`` satisfaction arrays, MASA uses:

    .. math::

       a \lor b \equiv 1 - (1-a)(1-b).

    Args:
        subformula_1: Left operand :math:`\Phi_1`.
        subformula_2: Right operand :math:`\Phi_2`.

    Attributes:
        subformula_1: Left operand.
        subformula_2: Right operand.
    """

    def __init__(self, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        """Bound is ``max(bound(subformula_1), bound(subformula_2))``."""
        return max(self.subformula_1.bound, self.subformula_2.bound)

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        """Compute the disjunction per state.

        Returns:
            Float64 array of shape ``(n_states,)`` in ``{0.0, 1.0}``.
        """
        return Neg(And(Neg(self.subformula_1), Neg(self.subformula_2))).sat(
            kernel, vec_label_fn, atom_dict
        )

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        """Compute the disjunction sequence using ``1 - (1-a)*(1-b)``.

        Returns:
            Float64 array of shape ``(max_k + 1, n_states)``.
        """
        seq1 = self.subformula_1._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        seq2 = self.subformula_2._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return 1.0 - (1.0 - seq1) * (1.0 - seq2)

class BoundedPCTLModelChecker:
    r"""Shared base for bounded PCTL model checkers.

    This class stores:

    - A bounded PCTL formula (:attr:`formula`).
    - A labeling function (:attr:`label_fn`) and a precomputed vectorized labeling
      matrix (:attr:`vec_label_fn`).
    - A transition representation in one of two forms:

      **Dense MDP kernel** (``mode="full"``)
        ``transition_matrix`` with shape ``(n_states, n_states, n_actions)``.
        The convention used throughout this module is **(next_state, state, action)**.

      **Compact successor kernel** (``mode="compact"``)
        ``successor_states`` with shape ``(K, n_states)`` and
        ``probabilities`` with shape ``(K, n_states, n_actions)``.

    Args:
        formula: The bounded PCTL formula to evaluate.
        label_fn: A :class:`~masa.common.label_fn.LabelFn` mapping
            ``state -> set[str]``.
        atomic_predicates: List of atom names (strings). These define the row
            ordering of :attr:`vec_label_fn`.
        transition_matrix: Optional dense MDP kernel of shape
            ``(n_states, n_states, n_actions)``.
        successor_states: Optional compact successor ids of shape ``(K, n_states)``.
        probabilities: Optional compact probabilities of shape
            ``(K, n_states, n_actions)``.

    Attributes:
        formula: Stored formula.
        label_fn: Stored labeling function.
        atomic_predicates: Atom vocabulary used to build :attr:`atom_dict`.
        atom_dict: Mapping from atom name to row index in :attr:`vec_label_fn`.
        mode: Either ``"full"`` or ``"compact"``.
        n_states: Number of states.
        n_actions: Number of actions.
        vec_label_fn: Float64 matrix of shape ``(n_atoms, n_states)``.

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
        """Build :attr:`vec_label_fn` from :attr:`label_fn`.

        Returns:
            Float64 array ``vec`` with shape ``(n_atoms, n_states)`` where
            ``vec[i, s] == 1.0`` iff atom ``i`` holds in state ``s``.
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

        Exactly one update mode should be used:

        - Dense update: provide ``transition_matrix`` with the same shape as the
          original.
        - Compact update: provide both ``successor_states`` and ``probabilities``
          with shapes consistent with the original compact representation.

        Args:
            transition_matrix: New dense MDP kernel ``(S, S, A)``.
            successor_states: New successor ids ``(K, S)``.
            probabilities: New successor probabilities ``(K, S, A)``.

        Raises:
            AssertionError: If shapes do not match the originally configured
                representation.
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
    r"""Exact model checker for bounded PCTL under a fixed policy.

    The exact checker collapses an MDP into a Markov chain by applying a
    stochastic policy, then evaluates the bounded formula on the resulting
    Markov chain using the formula's internal recurrences.

    - :meth:`check_state` returns per-state satisfaction for the policy-induced
      chain.
    - :meth:`check_state_action` returns a state-action value-like array derived
      from the formula probability sequence (using the vector at horizon ``B-1``).

    See Also:
        :class:`StatisticalModelChecker`: Sampling-based estimation of satisfaction.
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
        """Evaluate the stored formula on the policy-induced Markov chain.

        Args:
            key: Unused PRNG key (kept for API symmetry with
                :class:`StatisticalModelChecker`).
            policy: Stochastic policy probabilities, either:
                - shape ``(n_actions, n_states)`` (action-major), or
                - shape ``(n_states, n_actions)`` (state-major).

        Returns:
            Float64 array of shape ``(n_states,)`` in ``{0.0, 1.0}``.

        Raises:
            ValueError: If ``policy`` has an unexpected shape.
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

        This method:
        1) Builds the policy-induced Markov chain kernel,
        2) Computes the formula sequence up to bound ``B = formula.bound``,
        3) Uses the vector at time ``max(B-1, 0)`` as a value function,
        4) Computes one-step expectations under each action to produce ``Q``.

        Args:
            key: Unused PRNG key (kept for API symmetry with
                :class:`StatisticalModelChecker`).
            policy: Stochastic policy probabilities, either ``(A, S)`` or ``(S, A)``.

        Returns:
            Float64 array ``Q`` of shape ``(n_states, n_actions)`` where
            ``Q[s, a]`` is the expected satisfaction value after taking action
            ``a`` in state ``s`` and then following ``policy``.

        Raises:
            ValueError: If ``policy`` has an unexpected shape.
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
    r"""Statistical model checker (SMC) for bounded PCTL formulas.

    The SMC estimates satisfaction probabilities by Monte Carlo sampling of
    trajectories under a policy and comparing the estimated probability
    :math:`\hat{p}` to the formula’s threshold.

    Notes:
        - Pure state formulas (:class:`Truth`, :class:`Atom`, :class:`Neg`,
          :class:`And`, :class:`Or`, and :class:`Implies` if present) are evaluated
          exactly without sampling.
        - Nested probabilistic operators inside state formulas are not supported.

    Attributes:
        vec_label_fn_jax: JAX copy of :attr:`~BoundedPCTLModelChecker.vec_label_fn`.
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
        r"""Estimate whether ``state`` satisfies the formula under ``policy``.

        For probabilistic temporal formulas, this samples ``num_samples`` paths of
        length ``max_steps = max(1, formula.bound)`` and estimates

        .. math::

           \hat{p} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\{\pi_i \models \varphi\}.

        Args:
            key: PRNG key used for sampling.
            policy: Stochastic policy probabilities, either ``(A, S)`` or ``(S, A)``.
            state: Start state index.
            num_samples: Number of trajectories to sample.

        Returns:
            Scalar float64 JAX array equal to ``1.0`` if ``\hat{p} >= p`` else
            ``0.0``, where ``p`` is the formula’s probability threshold.
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
        r"""Estimate satisfaction when forcing the first action, then following ``policy``.

        The first transition uses the forced action ``action`` and subsequent
        steps follow the policy-induced kernel.

        Args:
            key: PRNG key used for sampling.
            policy: Stochastic policy probabilities, either ``(A, S)`` or ``(S, A)``.
            state: Start state index.
            action: Forced first action index.
            num_samples: Number of trajectories to sample.

        Returns:
            Scalar float64 JAX array equal to ``1.0`` if the estimated probability
            meets the formula's threshold, else ``0.0``.
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
