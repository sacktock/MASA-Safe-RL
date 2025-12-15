from masa.common.label_fn import LabelFn
from typing import Any, List, Dict
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

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
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
        max_k: int | None = None,
    ) -> np.ndarray:

        if max_k is None:
            max_k = self.bound

        v = self.sat(matrix, vec_label_fn, atom_dict)
        return np.repeat(v[None, :], max_k + 1, axis=0)

    def sat(
        self,
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        return self._prob_seq(matrix, vec_label_fn, atom_dict, max_k=self.bound)[
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
    def _next_prob_seq_core(
        matrix: jnp.ndarray,
        sub_seq: jnp.ndarray,
    ) -> jnp.ndarray:
        tail = matrix @ sub_seq.T
        tail = jnp.swapaxes(tail, 0, 1)

        zeros = jnp.zeros_like(tail[:1, :])
        seq = jnp.concatenate([zeros, tail], axis=0)
        return seq

    def _prob_seq(
        self,
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
        max_k: int | None = None,
    ) -> np.ndarray:

        if max_k is None:
            max_k = self.bound_param

        n_states = matrix.shape[0]
        if max_k == 0:
            return np.zeros((1, n_states), dtype=np.float32)

        seq = np.zeros((max_k + 1, n_states), dtype=np.float32)

        if max_k == 0:
            return seq

        sub_seq_np = self.subformula._prob_seq(
            matrix, vec_label_fn, atom_dict, max_k=max_k - 1
        )

        mat_j = jnp.asarray(matrix, dtype=jnp.float32)
        sub_seq_j = jnp.asarray(sub_seq_np, dtype=jnp.float32)

        seq_j = self._next_prob_seq_core(mat_j, sub_seq_j)
        return np.asarray(seq_j, dtype=np.float32)

    def sat(
        self,
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        probs = self._prob_seq(matrix, vec_label_fn, atom_dict, max_k=self.bound_param)[
            self.bound_param
        ]
        return (probs >= self.prob).astype(np.float32)

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
    @partial(jit, static_argnames=["max_k"])
    def _until_prob_seq_core(
        matrix: jnp.ndarray,
        sat1: jnp.ndarray,
        sat2: jnp.ndarray,
        max_k: int,
    ) -> jnp.ndarray:

        cont_mask = (1.0 - sat2) * sat1
        prob0 = sat2

        def body(prob, _):
            next_prob = sat2 + cont_mask * (matrix @ prob)
            return next_prob, next_prob

        _, probs_tail = jax.lax.scan(body, prob0, None, length=max_k)
        seq0 = prob0[None, :]
        seq = jnp.concatenate([seq0, probs_tail], axis=0)
        return seq

    def _prob_seq(
        self,
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
        max_k: int | None = None,
    ) -> np.ndarray:

        if max_k is None:
            max_k = self.bound_param

        n_states = matrix.shape[0]
        sat1 = self.subformula_1.sat(matrix, vec_label_fn, atom_dict).astype(np.float32)
        sat2 = self.subformula_2.sat(matrix, vec_label_fn, atom_dict).astype(np.float32)

        sat1_np = self.subformula_1.sat(
            matrix, vec_label_fn, atom_dict
        ).astype(np.float32)
        sat2_np = self.subformula_2.sat(
            matrix, vec_label_fn, atom_dict
        ).astype(np.float32)

        mat_j = jnp.asarray(matrix, dtype=jnp.float32)
        sat1_j = jnp.asarray(sat1_np, dtype=jnp.float32)
        sat2_j = jnp.asarray(sat2_np, dtype=jnp.float32)

        seq_j = self._until_prob_seq_core(mat_j, sat1_j, sat2_j, max_k=max_k)
        return np.asarray(seq_j, dtype=np.float32)

    def sat(
        self,
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        probs = self._prob_seq(matrix, vec_label_fn, atom_dict, max_k=self.bound_param)[
            self.bound_param
        ]
        return (probs >= self.prob).astype(np.float32)

class Always(BoundedPCTLFormula):

    def __init__(self, prob: float, bound: int, subformula: BoundedPCTLFormula):
        super().__init__()
        self.prob = float(prob)
        self.bound_param = int(bound)
        self.subformula = subformula

        self._inner = Neg(
            Until(
                prob=1.0 - self.prob,
                bound=self.bound_param,
                subformula_1=Truth(),
                subformula_2=Neg(self.subformula),
            )
        )

    @property
    def _bound(self) -> int:
        return self._inner.bound

    def _prob_seq(self, matrix, vec_label_fn, atom_dict, max_k=None):
        return self._inner._prob_seq(matrix, vec_label_fn, atom_dict, max_k)

    def sat(self, matrix, vec_label_fn, atom_dict):
        return self._inner.sat(matrix, vec_label_fn, atom_dict)

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

    def _prob_seq(self, matrix, vec_label_fn, atom_dict, max_k=None):
        return self._inner._prob_seq(matrix, vec_label_fn, atom_dict, max_k)

    def sat(self, matrix, vec_label_fn, atom_dict):
        return self._inner.sat(matrix, vec_label_fn, atom_dict)

class Truth(BoundedPCTLFormula):

    @property
    def _bound(self):
        return 0

    def sat(
        self,
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        n_states = matrix.shape[0]
        return np.ones(n_states, dtype=np.float32)

class Atom(BoundedPCTLFormula):

    def __init__(self, atom: str):
        super().__init__()
        self.atom = atom

    @property
    def _bound(self):
        return 0

    def sat(
        self,
        matrix: np.ndarray,
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

    def _prob_seq(self, matrix, vec_label_fn, atom_dict, max_k=None):
        seq = self.subformula._prob_seq(matrix, vec_label_fn, atom_dict, max_k)
        return 1.0 - seq

    def sat(
        self,
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        return 1.0 - self.subformula.sat(matrix, vec_label_fn, atom_dict)

class And(BoundedPCTLFormula):

    def __init__(self, subformula_1: BoundedPCTLFormula, subformula_2: BoundedPCTLFormula):
        super().__init__()
        self.subformula_1 = subformula_1
        self.subformula_2 = subformula_2

    @property
    def _bound(self):
        return max(self.subformula_1.bound, self.subformula_2.bound)

    def _prob_seq(self, matrix, vec_label_fn, atom_dict, max_k=None):
        seq1 = self.subformula_1._prob_seq(matrix, vec_label_fn, atom_dict, max_k)
        seq2 = self.subformula_2._prob_seq(matrix, vec_label_fn, atom_dict, max_k)
        return seq1 * seq2

    def sat(
        self,
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        return (
            self.subformula_1.sat(matrix, vec_label_fn, atom_dict)
            * self.subformula_2.sat(matrix, vec_label_fn, atom_dict)
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
        matrix: np.ndarray,
        vec_label_fn: np.ndarray,
        atom_dict: Dict[str, int],
    ) -> np.ndarray:
        return Neg(And(Neg(self.subformula_1), Neg(self.subformula_2))).sat(
            matrix, vec_label_fn, atom_dict
        )

    def _prob_seq(self, matrix, vec_label_fn, atom_dict, max_k=None):
        seq1 = self.subformula_1._prob_seq(matrix, vec_label_fn, atom_dict, max_k)
        seq2 = self.subformula_2._prob_seq(matrix, vec_label_fn, atom_dict, max_k)
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
        transition_matrix: np.ndarray,
        label_fn: LabelFn,
        atomic_predicates: List[str],
    ):
        assert len(transition_matrix.shape) == 3, \
            "transition_matrix must have shape (n_states, n_states, n_actions)"
        assert transition_matrix.shape[0] == transition_matrix.shape[1], \
            "transition_matrix must be square in first two dimensions"

        self.formula = formula
        # expected shape (n_states, n_states, n_actions)
        self.transition_matrix = np.asarray(transition_matrix, dtype=np.float32)
        self.n_states, _, self.n_actions = self.transition_matrix.shape
        self.label_fn = label_fn
        self.atomic_predicates = list(atomic_predicates)

        self.atom_dict: Dict[str, int] = {
            atom: i for i, atom in enumerate(self.atomic_predicates)
        }

        self.vec_label_fn = self._build_vec_label_fn()

    def _build_vec_label_fn(self) -> np.ndarray:

        n_states = self.transition_matrix.shape[0]
        n_atoms = len(self.atomic_predicates)
        vec = np.zeros((n_atoms, n_states), dtype=np.float32)

        for s in range(n_states):
            labels = self.label_fn(s) 

            for atom, idx in self.atom_dict.items():
                vec[idx, s] = 1.0 if atom in labels else 0.0

        return vec

    def update_transition_matrix(self, transition_matrix: np.ndarray):
        assert len(transition_matrix.shape) == 3, \
            "transition_matrix must have shape (n_states, n_states, n_actions)"
        assert transition_matrix.shape[0] == transition_matrix.shape[1], \
            "transition_matrix must be square in first two dimensions"
        assert transition_matrix.shape[0] == self.n_states and transition_matrix.shape[2] == self.n_actions, \
            f"transition_matrix does not match the original shape ({self.n_states}, {self.n_states}, {self.n_actions}), got f{transition_matrix.shape}"

        self.transition_matrix = np.asarray(transition_matrix, dtype=np.float32)

class ExactModelChecker(BoundedPCTLModelChecker):

    """
    Exact model checker: given a policy, collapses the MDP into a Markov chain
    and evaluates the PCTL formula on that chain.
    """

    def __init__(
        self,
        formula: BoundedPCTLFormula,
        transition_matrix: np.ndarray,
        label_fn: LabelFn,
        atomic_predicates: List[str],
    ):

        super().__init__(formula, transition_matrix, label_fn, atomic_predicates)

    def check_state(
        self,
        key: jax.Array, 
        policy: np.array
    ) -> np.ndarray:

        policy = np.asarray(policy, dtype=np.float32)

        tm = self.transition_matrix
        assert tm.ndim == 3, "transition_matrix must be (n_states, n_states, n_actions)"
        S, S2, A = tm.shape
        assert S == S2, "transition_matrix must be square in first two dimensions"

        if policy.shape == (A, S):
            m_pi = np.einsum('ija,ai->ij', tm, policy)
        elif policy.shape == (S, A):
            m_pi = np.einsum('ija,ia->ij', tm, policy)
        else:
            raise ValueError(
                f"Unexpected policy shape {policy.shape}; expected "
                f"(n_actions, n_states) or (n_states, n_actions)"
            )

        return self.formula.sat(m_pi, self.vec_label_fn, self.atom_dict)

    def check_state_action(
        self,
        key: jax.Array,
        policy: np.ndarray,
    ) -> np.ndarray:

        policy = np.asarray(policy, dtype=np.float32)

        tm = self.transition_matrix
        assert tm.ndim == 3, "transition_matrix must be (n_states, n_states, n_actions)"
        S, S2, A = tm.shape
        assert S == S2, "transition_matrix must be square in first two dimensions"

        if policy.shape == (A, S):
            m_pi = np.einsum('ija,ai->ij', tm, policy)
        elif policy.shape == (S, A):
            m_pi = np.einsum('ija,ia->ij', tm, policy)
        else:
            raise ValueError(
                f"Unexpected policy shape {policy.shape}; expected "
                f"(n_actions, n_states) or (n_states, n_actions)"
            )

        B = self.formula.bound

        seq = self.formula._prob_seq(m_pi, self.vec_label_fn, self.atom_dict, max_k=B)
        V_B_minus_1 = seq[max(B - 1, 0)]

        Q_B = np.einsum('nsa,n->sa', tm, V_B_minus_1)

        return Q_B

class StatisticalModelChecker(BoundedPCTLModelChecker):

    def __init__(
        self,
        formula: BoundedPCTLFormula,
        transition_matrix: np.ndarray,
        label_fn: LabelFn,
        atomic_predicates: List[str],
    ):
        super().__init__(formula, transition_matrix, label_fn, atomic_predicates)

        self.tm_jax = jnp.asarray(self.transition_matrix, dtype=jnp.float32)
        self.vec_label_fn_jax = jnp.asarray(self.vec_label_fn, dtype=jnp.float32)

        self.n_states, _, self.n_actions = self.tm_jax.shape

    def check_state(
        self,
        key: jax.Array, 
        policy: np.array,
        state: int,
        num_samples: int,
    ) -> np.ndarray:

        if not self._is_probabilistic_formula(self.formula):
            val = self._eval_state_formula_python(self.formula, state)
            return jnp.array(float(val), dtype=jnp.float32)

        start_state = int(state)
        num_samples = int(num_samples)
        max_steps = max(1, int(self.formula.bound))

        policy_probs = self._prepare_policy_probs_jax(
            self.n_states, self.n_actions, policy
        )

        m_pi = self._build_markov_chain_jax(self.tm_jax, policy_probs)
        prob_threshold = self._get_formula_prob(self.formula)

        path_satisfies_batch = jax.vmap(self._path_satisfies_single, in_axes=0)

        p_hat = self._estimate_prob(
            key=key,
            start_state=start_state,
            num_samples=num_samples,
            max_steps=max_steps,
            m_first=m_pi,
            m_rest=m_pi,
            formula=self.formula,
            vec_labels=self.vec_label_fn_jax,
            atom_dict=self.atom_dict
        )

        return jnp.where(p_hat >= prob_threshold, 1.0, 0.0).astype(jnp.float32)

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
            return jnp.array(float(val), dtype=jnp.float32)

        start_state = int(state)
        forced_action = int(action)
        num_samples = int(num_samples)
        max_steps = max(1, int(self.formula.bound))

        policy_probs = self._prepare_policy_probs_jax(
            self.n_states, self.n_actions, policy
        )

        m_a = self._build_forced_action_chain_jax(self.tm_jax, forced_action)
        m_pi = self._build_markov_chain_jax(self.tm_jax, policy_probs)

        prob_threshold = self._get_formula_prob(self.formula)

        p_hat = self._estimate_prob(
            key=key,
            start_state=start_state,
            num_samples=num_samples,
            max_steps=max_steps,
            m_first=m_a,
            m_rest=m_pi,
            formula=self.formula,
            vec_labels=self.vec_label_fn_jax,
            atom_dict=self.atom_dict
        )

        return jnp.where(p_hat >= prob_threshold, 1.0, 0.0).astype(jnp.float32)

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
        policy = jnp.asarray(policy, dtype=jnp.float32)
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
    @jit
    def _build_markov_chain_jax(tm: jnp.ndarray, policy_probs: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum('nsa,sa->ns', tm, policy_probs)

    @staticmethod
    @jit
    def _build_forced_action_chain_jax(tm: jnp.ndarray, action: int) -> jnp.ndarray:
        return tm[:, :, action]

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
    def _estimate_prob(
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

        states_batch = StatisticalModelChecker._sample_trajectories_jax(
            key=key,
            m_first=m_first,
            m_rest=m_rest,
            start_state=jnp.asarray(start_state, dtype=jnp.float32),
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
        return jnp.mean(sat_batch.astype(jnp.float32))

    @staticmethod
    @partial(jit, static_argnames=["num_samples", "max_steps"])
    def _sample_trajectories_jax(
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

        states0 = jnp.full((num_samples,), start_state, dtype=jnp.int32)

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
            next_states = next_states.astype(jnp.int32)

            return (next_states, key), next_states
            
        
        ts = jnp.arange(max_steps, dtype=jnp.int32)
        (final_states, _), states_seq = jax.lax.scan(body, (states0, key), ts)
        states_all = jnp.concatenate([states0[None, :], states_seq], axis=0)
        return states_all.T # expected output shape (N, T+1)
