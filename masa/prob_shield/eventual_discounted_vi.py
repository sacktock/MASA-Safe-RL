from __future__ import annotations
import jax
from jax import lax
from jax import jit
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

@jit
def ev_vi_one_step(
    q_vals: jnp.ndarray,
    successor_states_matrix: jnp.ndarray,
    probabilities: jnp.ndarray,
    unsafe_states: jnp.ndarray,
    gammas: jnp.ndarray,
):
    v = jnp.max(q_vals, axis=1)
    r_plus = unsafe_states + gammas * v
    r_next = jnp.take(r_plus, successor_states_matrix, axis=0)
    q_new = jnp.sum(r_next[..., None] * probabilities, axis=0)
    delta = jnp.max(jnp.abs(q_new - q_vals))
    return q_new, delta

def jax_ev_value_iteration(
    q_init: jnp.ndarray,
    successor_states_matrix: jnp.ndarray,
    probabilities: jnp.ndarray,
    unsafe_states: jnp.ndarray,
    gammas: jnp.ndarray,
    theta: float,
    max_steps: int,
) -> jnp.ndarray:

    def cond_fun(state):
        q_vals, delta, step = state
        return jnp.logical_and(delta > theta, step < max_steps)

    def body_fun(state):
        q_vals, _, step = state
        q_new, delta_new = ev_vi_one_step(
            q_vals,
            successor_states_matrix,
            probabilities,
            unsafe_states,
            gammas,
        )
        return (q_new, delta_new, step + 1)

    init_state = (
        q_init,
        jnp.inf,
        jnp.array(0, dtype=jnp.int32),
    )

    q_vals_final, _, _ = lax.while_loop(cond_fun, body_fun, init_state)
    return q_vals_final

def ev_value_iteration(
    successor_states_matrix: np.ndarray,
    probabilities: np.ndarray,
    label_fn: LabelFn,
    cost_fn: CostFn,
    theta: float = 1e-10,
    max_steps: int = 1000,
    gamma: float = 0.99,
) -> np.ndarray:

    n_states = successor_states_matrix.shape[1]
    n_actions = probabilities.shape[2]

    successor_states_abs = successor_states_matrix.copy()
    probabilities_abs = probabilities.copy()

    unsafe_states = np.zeros(n_states)
    gammas = np.ones(n_states)
    q_init = np.zeros((n_states, n_actions))

    # Turn unsafe states into absorbing states
    for s in range(n_states):
        if cost_fn(label_fn(s)): # absorbing unsafe states
            unsafe_states[s] = -1.0
            successor_states_abs[:, s] = s
            probabilities_abs[:, s, :] = 0.0
            probabilities_abs[0, s, :] = 1.0

    gammas[unsafe_states==1.0] = gamma

    successor_states_j = jnp.array(successor_states_abs, dtype=jnp.int64)
    probabilities_j = jnp.array(probabilities_abs, dtype=jnp.float64)
    unsafe_states_j = jnp.array(unsafe_states, dtype=jnp.float64)
    q_init_j = jnp.array(q_init, dtype=jnp.float64)
    gammas_j = jnp.array(gammas, dtype=jnp.float64)

    q_vals_j = jax_ev_value_iteration(
        q_init_j,
        successor_states_j,
        probabilities_j,
        unsafe_states_j,
        gammas_j,
        theta,
        max_steps,
    )

    return np.array(q_vals_j)
