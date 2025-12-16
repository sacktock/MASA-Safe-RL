from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
import numpy as np
from masa.common.wrappers import is_wrapped, get_wrapped
from masa.common.constraints import LTLSafetyEnv
from masa.common.ltl import DFA
from masa.common.constraints import CostFn
from masa.common.label_fn import LabelFn

def build_successor_states_matrix(
    env: gym.Env,
    label_fn: Optional[LabelFn] = None,
    cost_fn: Optional[CostFn] = None,
) -> Tuple[np.ndarray, np.ndarray, int, LabelFn, CostFn]:

    if not hasattr(env, "cost_fn") and cost_fn is None and env.cost_fn is None:
        raise AttributeError("Environment must define a `cost_fn` attribute.")
    cost_fn = cost_fn if cost_fn else env.cost_fn

    if not hasattr(env, "label_fn") and label_fn is None and env.label_fn is None:
        raise AttributeError("Environment must define a `label_fn` attribute.")
    label_fn = label_fn if label_fn else env.label_fn

    use_transition_matrix = bool(env.unwrapped.has_transition_matrix)
    use_successor_states_dict = not use_transition_matrix and bool(env.unwrapped.has_successor_states_dict) 

    if not (use_transition_matrix or use_successor_states_dict):
        raise ValueError(
            "Environment must expose either a transition matrix or a "
            "successor-states dictionary (or both), but neither was found."
        )

    if hasattr(env.unwrapped, "safe_end_component"):
        sec = env.unwrapped.safe_end_component
        try:
            iter(sec)
        except TypeError:
            raise TypeError(
                "`env.safe_end_component` must be an iterable (e.g. list, set, tuple) "
                f"but got type {type(sec).__name__}."
            )
        if len(sec) == 0:
            raise ValueError(
                "`env.safe_end_component` must be a non-empty iterable. "
                "The shielding procedure requires at least one absorbing safe state."
            )
    else:
        sec = None

    transition_matrix = env.unwrapped.get_transition_matrix() if use_transition_matrix else None
    successor_states, probs_dict = env.unwrapped.get_successor_states_dict() if use_successor_states_dict else (None, None)

    assert (transition_matrix is not None) or ((successor_states is not None) and (probs_dict is not None)),\
    "Something went wrong, you must provide either the transition matrix or both the successor states and probabilities for the environment"

    if not hasattr(env, "action_space"):
        raise AttributeError("Environment must have an `action_space` attribute.")
    if not isinstance(env.action_space, spaces.Discrete):
        raise TypeError(
            "ProbShieldWrapperDisc only supports environments with a "
            f"Discrete action space, got: {type(env.action_space).__name__}."
        )
    n_actions = env.action_space.n

    if is_wrapped(env, LTLSafetyEnv):
        from masa.common.constraints.ltl_safety import create_product_label_fn, product_cost_fn

        ltl_safety_env: LTLSafetyEnv | None = get_wrapped(env, LTLSafetyEnv)
        dfa = env._constraint.get_dfa()
        assert ltl_safety_env is not None

        if isinstance(ltl_safety_env._orig_obs_space, spaces.Discrete):
            n_states = ltl_safety_env._orig_obs_space.n
        else:
            if use_transition_matrix:
                n_states = transition_matrix.shape[0]
            if use_successor_states_dict:
                n_state = len(successor_states.keys())

        if use_transition_matrix:
            from masa.common.constraints.ltl_safety import create_product_transition_matrix
            transition_matrix = create_product_transition_matrix(
                n_states, n_actions, transition_matrix, dfa, label_fn
            )
        if use_successor_states_dict:
            from masa.common.constraints.ltl_safety import create_product_successor_states_and_probabilities
            successor_states, probs_dict = create_product_successor_states_and_probabilities(
                n_states, n_actions, successor_states, probs_dict, dfa, label_fn
            )
        if sec is not None:
            from masa.common.constraints.ltl_safety import create_product_safe_end_component
            sec = create_product_safe_end_component(
                n_states, n_actions, sec, dfa, label_fn,
            )
        label_fn = create_product_label_fn(n_states, dfa)
        cost_fn = product_cost_fn

    if not hasattr(env, "observation_space"):
        raise AttributeError("Environment must have an `observation_space` attribute.")
    if isinstance(env.observation_space, spaces.Discrete):
        n_states = env.observation_space.n
    else:
        if use_transition_matrix:
            n_states = transition_matrix.shape[0] # expected shape (n_states, n_states, n_actions)
        if use_successor_states_dict:
            n_states = len(successor_states.keys())

    if transition_matrix is not None:
        assert n_states == transition_matrix.shape[0], \
        "Something went wrong, the provided n_states does not equal the number of states in transition_matrix"
        f"Got n_states = {n_states} and transition_matrix.shape[0] == {transition_matrix.shape[0]}"
        assert n_actions == transition_matrix.shape[2], \
        "Something went wrong, the provided n_actions does not equal the number of states in transition_matrix"
        f"Got n_actions = {n_actions} and transition_matrix.shape[2] == {transition_matrix.shape[2]}"

    if successor_states is not None:
        base_states = sorted(successor_states.keys())
        assert n_states == len(base_states), \
        "Something went wrong, the provided n_states does not equal the numebr of states in successor_states "
        f"Got n_states = {n_states} and len(successor_states) = {len(base_states)}"

    print("Calculating the maximum number of successor states ...")

    if use_transition_matrix:
        probs = np.array(transition_matrix)
        reachable = np.any(probs > 0, axis=2)
        successor_counts = np.count_nonzero(reachable, axis=0)
        max_successors = successor_counts.max()

    if use_successor_states_dict:
        max_successors = 0
        for s in successor_states.keys():
            max_successors = max(max_successors, len(successor_states[s]))

    print(f"Calculated maximum number of successor states [{max_successors}] ...")

    successor_states_matrix = np.zeros((max_successors, n_states), dtype=np.int64)
    probabilities = np.zeros((max_successors, n_states, n_actions), dtype=np.float64)

    print("Building successor state matrix and probabilities ...")

    for s in range(n_states):
        if use_transition_matrix:
            mask = np.any(probs[:, s, :] > 0.0, axis=1)
            successors_s = np.nonzero(mask)[0]
            k = len(successors_s)
            successor_states_matrix[:k, s] = successors_s
            probabilities[:k, s, :] = probs[successors_s, s, :]

        if use_successor_states_dict:
            successors_s = np.array(successor_states[s], dtype=np.int64)#
            k = len(successors_s)
            successor_states_matrix[:k, s] = successors_s

            for a in range(n_actions):
                probabilities[:k, s, a] = np.array(
                    probs_dict[(s, a)],
                    dtype=np.float64
                )

    return successor_states_matrix, probabilities, max_successors, label_fn, cost_fn, sec