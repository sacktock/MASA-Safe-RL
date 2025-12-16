from __future__ import annotations
from typing import Any, Dict, List, Iterable, Callable, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from masa.common.label_fn import LabelFn
from masa.common.constraints import Constraint, BaseConstraintEnv
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

def create_product_safe_end_component(
    n_states: int,
    n_actions: int,
    sec: List[State],
    dfa: DFA,
    label_fn: LabelFn,
) -> List[ProdState]:

    product_sec = []

    aut_states = list(dfa.states)
    aut_index = {q: i for i, q in enumerate(aut_states)}
    accepting = list(dfa.accepting)

    for q_idx, q in enumerate(aut_states):
        for s in sec:
            labels = label_fn(s)
            q_next = dfa.transition(q, labels)
            if q_next in accepting:
                continue
        
            prod_state = q_idx * n_states + s
            product_sec.append(prod_state)

    return product_sec

def create_product_label_fn(
    n_states: int,
    dfa: DFA,
) -> Callable[[ProdState], Set[str]]:

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

    def __init__(self, dfa: DFA):
        self.cost_fn = dfa_to_costfn(dfa)

    def reset(self):
        self.safe = True
        self.step_cost = 0.0
        self.total_unsafe = 0.0
        self.cost_fn.reset()

    def update(self, labels: Iterable[str]):
        self.step_cost = self.cost_fn(labels)
        self.total_unsafe += float(self.step_cost >= 0.5)
        self.safe = self.safe and (not self.total_unsafe)

    def get_automaton_state(self):
        return self.cost_fn.automaton_state

    def get_dfa(self):
        return self.cost_fn.dfa

    def satisfied(self) -> bool:
        return self.safe

    def episode_metric(self) -> Dict[str, float]:
        return {"cum_unsafe": float(self.total_unsafe), "satisfied": float(self.satisfied())}

    def step_metric(self) -> Dict[str, float]:
        return {"cost": self.step_cost, "violation": float(self.step_cost >= 0.5)}

    @property
    def constraint_type(self) -> str:
        return "ltl_safety"


class LTLSafetyEnv(BaseConstraintEnv):

    def __init__(self, env: gym.Env, dfa: DFA = make_dummy_dfa(), **kw):
        super().__init__(env, LTLSafety(dfa=dfa), **kw)
        self._num_automaton_states = int(dfa.num_automaton_states)
        self._automaton_states_idx = {q: i for i, q in enumerate(dfa.states)}
        if self._num_automaton_states < 1:
            raise ValueError("dfa.num_automaton_states must be non-zero and positive")
        self._orig_obs_space = env.observation_space
        self.observation_space = self._make_augmented_obs_space(self._orig_obs_space)
        self._box_dtype = np.float32

    def _make_augmented_obs_space(self, orig: spaces.Space) -> spaces.Space:
        if isinstance(orig, spaces.Discrete):
            num_states = int(orig.n)
            aug = spaces.Discrete(num_states * self._num_automaton_states)
        elif isinstance(orig, spaces.Box):
            if orig.shape is None or len(orig.shape) != 1:
                raise TypeError(
                    f"LTLSafetyEnv only supports 1-D Box for augmentation; got shape {orig.shape}"
                )
            n = int(orig.shape[0])
            low = np.concatenate([orig.low.astype(self._box_dtype, copy=False),
                                  np.zeros(self._num_automaton_states, dtype=self._box_dtype)])
            high = np.concatenate([orig.high.astype(self._box_dtype, copy=False),
                                   np.ones(self._num_automaton_states, dtype=self._box_dtype)])
            aug = spaces.Box(low=low, high=high, dtype=self._box_dtype)
        elif isinstance(orig, spaces.Dict):
            automaton_space = spaces.Box(low=0.0, high=1.0, shape=(self._num_automaton_states,), dtype=self._box_dtype)
            new_spaces = dict(orig.spaces)
            new_spaces["automaton"] = automaton_space
            aug = spaces.Dict(new_spaces)
        else:
            raise TypeError(
                f"LTLSafetyEnv does not support observation space type {type(orig).__name__}. "
                "Supported: Discrete, 1-D Box, Dict."
            )
        return aug

    def _one_hot(self, q: int) -> np.ndarray:
        enc = np.zeros(self._num_automaton_states, dtype=self._box_dtype)
        if 0 <= q < self._num_automaton_states:
            enc[q] = 1
        return enc

    def _augment_obs(self, obs: Any) -> Any:
        q_idx = self._automaton_states_idx[self._constraint.get_automaton_state()]
        if isinstance(self.observation_space, spaces.Discrete):
            if not (isinstance(obs, (int, np.integer))):
                raise TypeError(f"Expected Discrete obs as int, got {type(obs).__name__}")
            return self._orig_obs_space.n * int(q_idx) + int(obs)
        if isinstance(self.observation_space, spaces.Box):
            if not isinstance(obs, np.ndarray):
                obs = np.asarray(obs, dtype=self._box_dtype)
            if obs.ndim != 1:
                raise TypeError(f"Expected 1-D Box observation, got shape {getattr(obs, 'shape', None)}")
            enc = self._one_hot(q_idx, dtype=self._box_dtype)
            return np.concatenate([obs.astype(self._box_dtype, copy=False), enc], axis=0)
        if isinstance(self.observation_space, spaces.Dict):
            out = dict(obs) if isinstance(obs, dict) else {}
            out["automaton"] = self._one_hot(q_idx, dtype=np.float32)
            return out

        raise RuntimeError(f"Unexpected observation space type {self.observation_space}")

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
    

        