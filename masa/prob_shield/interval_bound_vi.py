from jax import jit
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from masa.common.constraints.base import CostFn
from masa.common.label_fn import LabelFn
from typing import Tuple, List

jax.config.update("jax_enable_x64", True)

@jit
def vi_one_step(
    v_inf: jnp.ndarray, 
    v_sup: jnp.ndarray, 
    successor_states_matrix: jnp.ndarray,
    probabilities: jnp.ndarray
)-> Tuple[jnp.ndarray, jnp.ndarray, float, int]:
    """
    v_inf, v_sup: (n_states,)
    successor_states_matrix: (max_successors, n_states) int
    probabilities: (max_successors, n_states, n_actions)
    """

    next_v_inf = jnp.take(v_inf, successor_states_matrix, axis=0)
    next_v_sup = jnp.take(v_sup, successor_states_matrix, axis=0)

    exp_inf = jnp.sum(next_v_inf[..., None] * probabilities, axis=0)
    exp_sup = jnp.sum(next_v_sup[..., None] * probabilities, axis=0)

    v_inf_new = jnp.min(exp_inf, axis=-1)
    v_sup_new = jnp.min(exp_sup, axis=-1)

    v_inf_new = jnp.clip(v_inf_new, 0.0, 1.0)
    v_sup_new = jnp.clip(v_sup_new, 0.0, 1.0)

    delta = jnp.max(jnp.abs(v_sup_new - v_inf_new))

    return v_inf_new, v_sup_new, delta

def jax_interval_bound_value_iteration(
    v_inf_init: jnp.ndarray,
    v_sup_init: jnp.ndarray,
    successor_states_matrix: jnp.ndarray,
    probabilities: jnp.ndarray,
    theta: float,
    max_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, int]:

    v_inf = v_inf_init
    v_sup = v_sup_init
    delta = jnp.inf
    steps = 0

    def cond(carry):
        v_inf, v_sup, delta, steps = carry
        return jnp.logical_and(steps < max_steps, delta > theta)

    def body(carry):
        v_inf, v_sup, _, steps = carry
        v_inf_new, v_sup_new, delta_new = vi_one_step(
            v_inf, v_sup, successor_states_matrix, probabilities
        )
        return v_inf_new, v_sup_new, delta_new, steps + 1

    v_inf, v_sup, delta, steps = lax.while_loop(
        cond,
        body,
        (v_inf, v_sup, delta, steps)
    )
    return v_inf, v_sup, delta, steps

def interval_bound_value_iteration(
    successor_states_matrix: np.ndarray,
    probabilities: np.ndarray,
    label_fn: LabelFn,
    cost_fn: CostFn,
    sec: List[int],
    theta: float = 1e-10,
    max_steps: int = 1000,
)-> Tuple[np.ndarray, np.ndarray, float, int]:

    n_states = successor_states_matrix.shape[1] # expected shape (max_successors, n_states)

    # Copy to edit in-place
    successor_states_abs = successor_states_matrix.copy()
    probabilities_abs = probabilities.copy()

    print("Initializing value iteration ...")

    # Initial bounds
    v_inf = np.zeros(n_states, dtype=np.float64)
    v_sup = np.ones(n_states, dtype=np.float64)

    for s in range(n_states):
        absorbing = False
        if cost_fn(label_fn(s)):
            v_inf[s] = 1.0
            absorbing = True 
        if s in sec:
            v_sup[s] = 0.0
            absorbing = True 
        if absorbing:
            successor_states_abs[:, s] = s
            probabilities_abs[:, s, :] = 0.0
            probabilities_abs[0, s, :] = 1.0

    v_inf_j = jnp.array(v_inf)
    v_sup_j = jnp.array(v_sup)
    successor_states_j = jnp.array(successor_states_abs, dtype=jnp.int64)
    probabilities_j = jnp.array(probabilities_abs, dtype=jnp.float64)
    
    v_inf_j, v_sup_j, delta, steps = jax_interval_bound_value_iteration(
        v_inf_j,
        v_sup_j,
        successor_states_j,
        probabilities_j,
        theta,
        max_steps,
    )

    print("Completed value iteration ...")

    v_inf_final = np.array(v_inf_j)
    v_sup_final = np.array(v_sup_j)

    return v_inf_final, v_sup_final, delta, int(steps)