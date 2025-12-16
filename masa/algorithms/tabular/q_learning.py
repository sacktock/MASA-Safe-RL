from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable, Dict
from masa.common.metrics import TrainLogger
from masa.algorithms.tabular.base import TabularAlgorithm
from masa.common.ltl import DFACostFn, DFA
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial


class QL(TabularAlgorithm):

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
        alpha: float = 0.1,
        gamma: float = 0.9,
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
        )

        assert exploration in ["boltzmann", "epsilon_greedy"], f"Unsupported exploration type: {exploration}"
        assert epsilon_decay in ["linear"], f"Unsupported epsilon decay schedule: {epsilon_decay}"

        self.n_states = self.observation_space.n
        self.n_actions = self.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.boltzmann_temp = boltzmann_temp
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_frames = epsilon_decay_frames

        self._setup_q_table()
        self._setup_buffer()
        self._setup_decay_schedule()
        
    def _setup_q_table(self):
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def _setup_buffer(self):
        """Setup a simple buffer that stores one tuple to update"""
        self.buffer = []

    def _setup_decay_schedule(self):
        self._epsilon = self.initial_epsilon
        self._step = 0
        if self.epsilon_decay == "linear":
            self._epsilon_decay_schedule = lambda step: self.final_epsilon + \
                (self.initial_epsilon - self.final_epsilon) * \
                max(0, (self.epsilon_decay_frames - step) / self.epsilon_decay_frames)
        else:
            raise NotImplementedError(f"epsilon decay schedule: {self.epsilon_decay}")

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q table with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, action, reward, _, _, next_state, terminal) in self.buffer:

            current = self.Q[next_state]
            self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
            + self.alpha * (reward + (1 - terminal) * self.gamma * np.max(current))

        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon_greedy":
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
        self, state: int, action: int, reward: float, next_state: int, terminated: bool, info: Dict[str, Any], cost_fn: DFACostFn,
    ) -> List[Tuple[int, int, float, float, bool, int, bool]]:

        dfa: DFA = cost_fn.dfa
        counter_fac_exp = []

        base_n = self.n_states // dfa.num_automaton_states
        labels = info.get("labels", set())

        base_state = state % base_n
        base_next_state = next_state % base_n

        for i, automaton_state in enumerate(dfa.states):
            next_automaton_state = dfa.transition(automaton_state, labels)
            j = dfa.states.index(next_automaton_state)

            cost_cf = cost_fn.cost(automaton_state, labels)
            violation_cf = bool(next_automaton_state in dfa.accepting)

            cf_state = base_state + base_n * i
            cf_next_state = base_next_state + base_n * j

            counter_fac_exp.append(
                (
                    cf_state,
                    action,
                    reward,
                    cost_cf,
                    violation_cf,
                    cf_next_state,
                    terminated,
                )
            )

        return counter_fac_exp


    def act(self, key, obs, deterministic=False):
        if deterministic:
            action = self.select_action(self.Q[obs])
        else:
            action = self.sample_action(
                key, 
                jnp.asarray(self.Q[obs], dtype=jnp.float32), 
                self.boltzmann_temp, 
                self._epsilon, 
                exploration=self.exploration
            )
        return self.prepare_act(action)
            
    def prepare_act(self, act: Any):
        return int(np.asarray(act).item())

    @staticmethod
    def select_action(q_values: np.ndarray):
        return np.argmax(q_values)

    @staticmethod
    @partial(jit, static_argnames=["exploration"])
    def sample_action(key, q_values, tmp, eps, exploration="boltzmann"):
        if exploration == "boltzmann":
            scaled_q = q_values - jnp.max(q_values)
            exp = jnp.exp(scaled_q / tmp)
            probs = exp / (jnp.sum(exp) + 1e-6)
        if exploration == "epsilon_greedy":
            probs = jnp.zeros(q_values.shape[0])
            probs = probs.at[jnp.argmax(q_values)].set(1.0 - eps)
            probs += (eps / q_values.shape[0])
        return jr.choice(key, q_values.shape[0], p=probs)

    @property
    def train_ratio(self):
        return 1