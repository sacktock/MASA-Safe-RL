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
    if isinstance(kernel, tuple):
        succ, p = kernel
        return succ.shape[1]
    return kernel.shape[1]

class BoundedPCTLFormula:
    def __init__(self) -> None:
        pass

    @property
    def _bound(self) -> int:
        raise NotImplementedError

    @property
    def bound(self) -> int:
        return self._bound

    def _prob_seq(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
        max_k: int | None = None,
    ) -> np.ndarray:

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
        return self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound)[
            self.bound
        ]

class Next(BoundedPCTLFormula):

    def __init__(self, prob: float, subformula: BoundedPCTLFormula):
        super().__init__()
        self.prob = prob
        self.subformula = subformula
        self.bound_param = 1

    @property
    def _bound(self):
        return self.bound_param + self.subformula.bound

    @staticmethod
    @jit
    def _next_prob_seq_core_dense(m: jnp.ndarray, sub_seq: jnp.ndarray) -> jnp.ndarray:
        tail = (m.T @ sub_seq.T)
        tail = jnp.swapaxes(tail, 0, 1)
        zeros = jnp.zeros_like(tail[:1, :])
        return jnp.concatenate([zeros, tail], axis=0)

    @staticmethod
    @partial(jit, static_argnames=("K",))
    def _next_prob_seq_core_compact(succ: jnp.ndarray, p: jnp.ndarray, sub_seq: jnp.ndarray, K: int) -> jnp.ndarray:
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
        probs = self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound_param)[
            self.bound_param
        ]
        return (probs >= self.prob).astype(np.float64)

class Until(BoundedPCTLFormula):

    def __init__(self, prob: float, bound, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.prob = prob
        self.bound_param = int(bound)
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        return self.bound_param + self.subformula_1.bound + self.subformula_2.bound

    @staticmethod
    @partial(jit, static_argnames=("max_k",))
    def _until_prob_seq_core_dense(m: jnp.ndarray, sat1: jnp.ndarray, sat2: jnp.ndarray, max_k: int) -> jnp.ndarray:
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
        probs = self._prob_seq(kernel, vec_label_fn, atom_dict, max_k=self.bound_param)[
            self.bound_param
        ]
        return (probs >= self.prob).astype(np.float64)

class Always(BoundedPCTLFormula):

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
        return self._inner.bound

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        return self._inner._prob_seq(kernel, vec_label_fn, atom_dict, max_k)

    def sat(self, kernel, vec_label_fn, atom_dict):
        return self._inner.sat(kernel, vec_label_fn, atom_dict)

class Eventually(BoundedPCTLFormula):

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
        return self._inner.bound

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        return self._inner._prob_seq(kernel, vec_label_fn, atom_dict, max_k)

    def sat(self, kernel, vec_label_fn, atom_dict):
        return self._inner.sat(kernel, vec_label_fn, atom_dict)

class Truth(BoundedPCTLFormula):

    @property
    def _bound(self):
        return 0

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        n_states = kernel_n_states(kernel)
        return np.ones(n_states, dtype=np.float64)

class Atom(BoundedPCTLFormula):

    def __init__(self, atom: str):
        super().__init__()
        self.atom = atom

    @property
    def _bound(self):
        return 0

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        return vec_label_fn[atom_dict[self.atom]]

class Neg(BoundedPCTLFormula):

    def __init__(self, subformula: BoundedPCTLFormula):
        super().__init__()
        self.subformula = subformula

    @property
    def _bound(self):
        return self.subformula.bound

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        seq = self.subformula._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return 1.0 - seq

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        return 1.0 - self.subformula.sat(kernel, vec_label_fn, atom_dict)

class And(BoundedPCTLFormula):

    def __init__(self, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        return max(self.subformula_1.bound, self.subformula_2.bound)

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        seq1 = self.subformula_1._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        seq2 = self.subformula_2._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return seq1 * seq2

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        return (
            self.subformula_1.sat(kernel, vec_label_fn, atom_dict)
            * self.subformula_2.sat(kernel, vec_label_fn, atom_dict)
        )

class Or(BoundedPCTLFormula):

    def __init__(self, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        return max(self.subformula_1.bound, self.subformula_2.bound)

    def sat(
        self,
        kernel: Kernel,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        return Neg(And(Neg(self.subformula_1), Neg(self.subformula_2))).sat(
            kernel, vec_label_fn, atom_dict
        )

    def _prob_seq(self, kernel, vec_label_fn, atom_dict, max_k=None):
        seq1 = self.subformula_1._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        seq2 = self.subformula_2._prob_seq(kernel, vec_label_fn, atom_dict, max_k)
        return 1.0 - (1.0 - seq1) * (1.0 - seq2)

class BoundedPCTLModelChecker:

    """
    transition_matrix: shape (n_states, n_states, n_actions)
    label_fn: LabelFn describing which atoms hold in which states.
    atomic_predicates: list of atomic proposition names (strings).
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

    """
    Exact model checker: given a policy, collapses the MDP into a Markov chain
    and evaluates the PCTL formula on that chain.
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
        return isinstance(formula, (Next, Until, Eventually, Always))

    def _get_formula_prob(self, formula: BoundedPCTLFormula) -> float:
        if isinstance(formula, (Next, Until, Eventually, Always)):
            return float(formula.prob)
        raise ValueError(
            "Trying to get probability threshold from non-probabilistic formula."
        )

    def _eval_state_formula_python(self, formula: BoundedPCTLFormula, state: int) -> bool:
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
    def _prepare_policy_probs_jax(n_states: int, n_actions: int, policy: jnp.ndarray) -> jnp.ndarray:
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
