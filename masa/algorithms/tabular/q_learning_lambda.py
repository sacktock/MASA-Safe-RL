from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable
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

class QL_Lambda(QL):

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
        cost_lambda: float = 1.0,
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
            alpha=alpha,
            gamma=gamma,
            exploration=exploration,
            boltzmann_temp=boltzmann_temp,
            initial_epsilon=initial_epsilon,
            final_epsilon=final_epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_decay_frames=epsilon_decay_frames,
        )

        self.cost_lambda = cost_lambda

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q table with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, action, reward, cost, violation, next_state, terminal) in self.buffer:

            penalty = -self.cost_lambda * cost

            current = self.Q[next_state]
            self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
            + self.alpha * ((reward + penalty) + (1 - violation) * (1 - terminal) * self.gamma * np.max(current))

        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            logger.add("train/stats", {"cost_lambda": self.cost_lambda})
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
