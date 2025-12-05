from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable, Dict
from masa.common.metrics import TrainLogger
from masa.algorithms.tabular.q_learning import QL
from masa.common.ltl import DFACostFn, DFA
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial
from tqdm import tqdm

def ev_value_iteration(n_states: int, n_actions: int, transition_matrix: np.ndarray, reward_array: np.ndarray, gamma: float= 0.99, steps: int = 100):

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    gamma_array = np.ones_like(reward_array, dtype=np.float32)
    gamma_array[reward_array != 0.0] = gamma

    print("Running value iteration ...")

    for i in tqdm(range(steps)):
        r_plus = reward_array + gamma_array * np.max(Q, axis=1)
        Q = np.tensordot(r_plus, transition_matrix, axes=(0, 0))

    return Q

def calulate_sat_relation(n_states: int, n_actions: int, transition_matrix: np.ndarray, B: np.ndarray, unsafe_array: np.ndarray, horizon: int):

    S = np.zeros((horizon, n_states, n_actions), dtype=np.float32)
    policy = np.argmax(B, axis=1).astype(np.int64)

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            unsafe_plus = unsafe_array.astype(np.float32)
        else:
            unsafe_plus = unsafe_array + (1.0 - unsafe_array) * S[t + 1, np.arange(n_states), policy]
        S[t] = np.tensordot(unsafe_plus, transition_matrix, axes=(0, 0))

    return S

class RECREG(QL):

    def __init__(
        self,
        env: gym.Env,
        tensorboard_logdir: Optional[str] = None,
        seed: Optional[int] = None,
        monitor: bool = True,
        device: str = "auto",
        verbose: int = 0,
        env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None, 
        task_alpha: float = 0.1,
        task_gamma: float = 0.9,
        safe_alpha: float = 0.1,
        safe_gamma: float = 0.9,
        model_impl: str = 'model-based',
        model_checking: str = 'sample',
        samples: int = 512,
        horizon: int = 10,
        step_wise_prob: float = 0.99,
        model_prior: str = 'identity',
        exploration: str = 'boltzmann',
        boltzmann_temp: float = 0.05,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        epsilon_decay: str = 'linear',
        epsilon_decay_frames: int = 10000, 
    ):

        super().__init__(
            env, 
            tensorboard_logdir=tensorboard_logdir,
            seed=seed,
            monitor=monitor,
            device=device,
            verbose=verbose,
            env_fn=env_fn,
            eval_env=eval_env,
            alpha=task_alpha,
            gamma=task_gamma,
            exploration=exploration,
            boltzmann_temp=boltzmann_temp,
            initial_epsilon=initial_epsilon,
            final_epsilon=final_epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_decay_frames=epsilon_decay_frames,
        )

        assert model_impl in ["exact", "model-based", "model-free"], f"Unsupported model implementation {model_impl}"
        assert model_checking in ["exact", "sample", "none"], f"Unsupported model-checking type {rollout}"
        assert not model_checking == "none" or model_impl == "model-free", f"Must specify a model checking type when using model_impl: {model_impl}"
        assert samples > 0, "Number of samples must be greater than 0"
        assert model_prior in ["identity", "uniform"], f'Prior for transition model must be one of: "identity" or "uniform"'

        self.safe_alpha = safe_alpha
        self.safe_gamma = safe_gamma
        self.model_impl = model_impl
        self.model_checking = model_checking
        self.samples = samples
        self.horizon = horizon
        self.step_wise_prob = step_wise_prob
        self.model_prior = model_prior

        self._setup_models()

    def _setup_models(self):
        if self.model_impl == "exact":
            assert self.env.has_transition_matrix(), f"Cannot instantiate RecReg with 'exact' model implementation for an environment that does not expose its transtion matrix"
            raise NotImplementedError("TODO: implelemt penalty_array and unsafe_array")
            self.B = ev_value_iteration(self.n_states, self.n_actions, self.env.get_transition_matrix(), penalty_array, gamma=self.safe_gamma, steps=1000)
            self.S = calulate_sat_relation(self.n_states, self.n_actions, self.env.get_transition_matrix(), self.B, unsafe_array, self.horizon)
        elif self.model_impl == "model-based":
            self.B = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
            if self.model_prior == "identity":
                self.pseudo_counts = np.tile(np.eye(self.n_states, dtype=np.float32)[:, :, np.newaxis], [1, 1, self.n_actions])
            elif self.model_prior == "uniform":
                self.pseudo_counts = (np.ones((self.n_states, self.n_states, self.n_actions), dtype=np.float32) / float(self.n_states))
            else:
                raise NotImplementedError(f"Unexpected model_prior: {self.model_prior}")
        elif self.model_impl == "model-free":
            self.B = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
            self.S = np.zeros((self.horizon, self.n_states, self.n_actions), dtype=np.float32)
        else:
            raise NotImplementedError(f"Unexpected model_impl: {self.model_impl}")

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q table with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, action, reward, cost, violation, next_state, terminal) in self.buffer:

            penalty = -float(cost)

            current = self.Q[next_state]
            if self.S[0, state, action] <= self.step_wise_prob:
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
                + self.alpha * (reward + (1 - terminal) * self.gamma * np.max(current))
            else:
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action]

            safe_gamma = self.safe_gamma if violation else 1.0
            current = self.B[next_state]
            self.B[state, action] = (1 - self.safe_alpha) * self.B[state, action] \
            + self.safe_alpha * (penalty + (1 - terminal) * safe_gamma * np.max(current))

            if self.model_impl == "model-based":
                self.pseudo_counts[next_state, state, action] += 1.0

            if self.model_impl == "model-free":
                current = self.B[next_state]
                for t in reversed(range(self.horizon)):
                    if t == self.horizon - 1:
                        next_S = min(1.0, float(violation))
                    else:
                        next_S = min(1.0, float(violation) + (1 - terminal) * self.S[t+1, next_state, np.argmax(current)])
                    self.S[t, state, action] = (1 - self.safe_alpha) * self.S[t, state, action] + self.safe_alpha * next_S

        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            logger.add("train/stats", {"safe_alpha": self.safe_alpha})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon-greedy":
                logger.add("train/stats", {"epsilon": self._epsilon})

    def rollout(self, step: int, logger: Optional[TrainLogger] = None):

        self.key, subkey = jr.split(self.key)
        action = self.act(subkey, self._last_obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        if hasattr(self.env, "cost_fn") and isinstance(self.env.cost_fn, DFACostFn):
            # if the cost function associated with the constraint is a DFA, by default use counter factual experiences
            self.buffer += self.generate_counter_factuals(
                self._last_obs, action, reward, next_obs, terminated, info, getattr(self.env._constraint, "cost_fn", None)
            )
        else:
            cost = info["constraint"]["step"].get("cost", 0.0)
            violation = info["constraint"]["step"].get("violation", False)
            self.buffer.append((self._last_obs, action, reward, cost, violation, next_obs, terminated))

        if terminated or truncated:
            self._last_obs, _ = self.env.reset()
        else:
            self._last_obs = next_obs

        self._step += 1
        self._epsilon = self._epsilon_decay_schedule(self._step)

        if logger:
            logger.add("train/rollout", info)

    def generate_counter_factuals(
        self, obs: int, action: int, reward: float, next_obs: int, terminated: bool, info: Dict[str, Any], cost_fn: DFACostFn
    ) -> List[Tuple[int, int, float, int, bool]]:

        dfa: DFA = cost_fn.dfa
        counter_fac_exp = []

        _n = self.n_states // dfa.num_automaton_states

        _labels = info.get("labels", set())

        _obs = obs % _n
        _next_obs = next_obs % _n

        for _i, counter_fac_automaton_state in enumerate(dfa.states):
            next_counter_fac_automaton_state = dfa.transition(counter_fac_automaton_state, _labels)
            _j = dfa.states.index(next_counter_fac_automaton_state)
            cost = cost_fn.cost(counter_fac_automaton_state, _labels)
            violation = bool(next_counter_fac_automaton_state in dfa.accepting)
            counter_fac_exp.append(
                (_obs + _n * _i,
                action,
                reward,
                cost,
                violation,
                _next_obs + _n * _j,
                terminated,)
            )

        return counter_fac_exp

    def act(self, key, obs, deterministic=False):
        action = self._act(key, obs, self.Q[obs], deterministic=deterministic)
        action = np.array(action, dtype=np.int64)
        if self.model_impl in ["exact", "model-free"]:
            if self.S[0, obs, action] <= self.step_wise_prob:
                return self.prepare_act(action)
            else:
                action = self._act(key, obs, self.B[obs], deterministic=deterministic)
                return self.prepare_act(action)
        elif self.model_impl == "model-based":
            raise NotImplementedError("TODO: implement exact or numerical model checking for model-based transition model")
        else:
            raise NotImplementedError(f"Unexpected model_impl: {self.model_impl}")
        
    def _act(self, key, obs, Q, deterministic=False):
        if deterministic:
            return self.select_action(Q)
        else:
            return self.sample_action(
                key, jnp.asarray(Q, dtype=jnp.float32), self.boltzmann_temp, self._epsilon, exploration=self.exploration
            )


