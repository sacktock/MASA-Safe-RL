from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable, Dict, Tuple, List
from masa.common.metrics import TrainLogger
from masa.algorithms.tabular.q_learning import QL
from masa.common.ltl import DFACostFn
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm

def q_to_boltzmann_policy(Q: np.ndarray, tau: float) -> np.ndarray:
    prefs = Q / tau
    max_prefs = np.max(prefs, axis=1, keepdims=True)
    exp_prefs = np.exp(prefs - max_prefs)
    return exp_prefs / np.sum(exp_prefs, axis=1, keepdims=True) 

def q_to_argmax_policy(Q):
    n_states, n_actions = Q.shape
    max_vals = np.max(Q, axis=1, keepdims=True)
    greedy_mask = (Q == max_vals).astype(float)
    greedy_policy = greedy_mask / greedy_mask.sum(axis=1, keepdims=True)
    return greedy_policy

class RECOVERY_RL(QL):

    def __init__(
        self,
        env: gym.Env,
        tensorboard_logdir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        monitor: bool = True,
        device: str = "auto",
        verbose: int = 0,
        env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None, 
        task_alpha: float = 0.1,
        task_gamma: float = 0.9,
        risk_alpha: float = 0.1,
        risk_gamma: float = 0.99,
        step_wise_risk: float = 0.01,
        exploration: str = "boltzmann",
        boltzmann_temp: float = 0.05,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        epsilon_decay: str = "linear",
        epsilon_decay_frames: int = 10000, 
    ):

        super().__init__(
            env, 
            tensorboard_logdir=tensorboard_logdir,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
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

        self.risk_alpha = risk_alpha
        self.risk_gamma = risk_gamma
        self.step_wise_risk = step_wise_risk

        self._setup_models()

    def _setup_models(self):
        self.Q_rec = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.Q_risk = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def train(self, *args, **kwargs):
        self._overrides = []
        super().train(*args, **kwargs)

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q table with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, action, task_action, reward, cost, violation, next_state, terminal, info) in self.buffer:

            if hasattr(self.env, "cost_fn") and isinstance(self.env.cost_fn, DFACostFn):
                cf_exp = self.generate_counter_factuals(
                    state, action, reward, next_state, terminal, info, getattr(self.env, "cost_fn", None)
                )
            else:
                cf_exp = [(state, action, reward, cost, violation, next_state, terminal)]

            for exp in cf_exp:
                cf_state, cf_act, cf_rew, cf_cost, cf_viol, cf_next_state, cf_term = exp

                self.Q[cf_state, task_action] = (
                    (1 - self.alpha) * self.Q[cf_state, task_action] \
                    + self.alpha * (cf_rew + (1 - cf_term) * (1 - cf_viol) * self.gamma * np.max(self.Q[cf_next_state]))
                )
                

                self.Q_rec[cf_state, cf_act] = (
                    (1 - self.risk_alpha) * self.Q_rec[cf_state, cf_act] \
                    + self.risk_alpha * (float(cf_viol) + (1 - cf_term) * (1 - cf_viol) * self.risk_gamma * np.min(self.Q_rec[cf_next_state]))
                )
                
                
                if self.exploration == "boltzmann":
                    self.task_policy = q_to_boltzmann_policy(self.Q, self.boltzmann_temp)
                elif self.exploration == "epsilon_greedy":
                    self.task_policy = q_to_argmax_policy(self.Q)
                else:
                    raise NotImplementedError(f"Unexpected exploration: {self.exploration}")

                self.Q_risk[cf_state, cf_act] =  (
                    (1 - self.risk_alpha) * self.Q_risk[cf_state, cf_act] \
                    + self.risk_alpha * (float(cf_viol) + (1 - cf_term) * (1 - cf_viol) * self.risk_gamma * (self.task_policy[cf_next_state] * self.Q_risk[cf_next_state]).sum())
                )
                    
        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            logger.add("train/stats", {"risk_alpha": self.risk_alpha})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon_greedy":
                logger.add("train/stats", {"epsilon": self._epsilon})
            

    def rollout(self, step: int, logger: Optional[TrainLogger] = None):

        self.key, subkey = jr.split(self.key)
        action, task_action, override = self.act_override(subkey, self._last_obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self._overrides.append(override)

        cost = info["constraint"]["step"].get("cost", 0.0)
        violation = info["constraint"]["step"].get("violation", False)

        self.buffer.append(
            (self._last_obs, action, task_action, reward, cost, violation, next_obs, terminated, info)
        )

        if terminated or truncated:
            self._last_obs, _ = self.env.reset()
            info["constraint"]["episode"].update({"override_rate": float(np.mean(self._overrides))})
            self._overrides.clear()
        else:
            self._last_obs = next_obs

        self._step += 1
        self._epsilon = self._epsilon_decay_schedule(self._step)

        if logger:
            logger.add("train/rollout", info)

    def eval(self, num_episodes: int, seed: Optional[int] = None, logger: Optional[TrainLogger] = None) -> List[float]:
        eval_env = self._get_eval_env()
        base = 0 if self.seed is None else int(self.seed)
        eval_seed = base + 10_000 if seed is None else int(seed) + 10_000
        eval_key = jr.PRNGKey(eval_seed)
        returns = []
        overrides = []

        with tqdm(
            total=num_episodes,
            desc="evaluation",
            position=1,
            leave=False,
            dynamic_ncols=True,
            colour="yellow"
        ) as pbar:

            for ep in range(num_episodes):
                obs, info = eval_env.reset(seed=eval_seed + ep)
                done = False
                ret = 0.0
                while not done:
                    eval_key, subkey = jr.split(eval_key)
                    action, _, override = self.act_override(subkey, obs, deterministic=False)
                    overrides.append(override)
                    obs, rew, terminated, truncated, info = eval_env.step(action)
                    ret += float(rew)
                    done = terminated or truncated

                info["constraint"]["episode"].update({"override_rate": float(np.mean(overrides))})
                overrides.clear()

                returns.append(ret)
                pbar.update(1)

                if logger is not None:
                    logger.add("eval/rollout", info)

        return returns

    def act_override(self, key, obs, deterministic=False):
        key, key1, key2 = jr.split(key, 3)
        action = task_action = self._act(key1, obs, self.Q, deterministic, self.boltzmann_temp, self._epsilon)
        override = self.Q_risk[obs, action] > self.step_wise_risk
        if override:
            action = self._act(key2, obs, self.Q_rec, deterministic, self.boltzmann_temp, 0.01)
        return self.prepare_act(action), self.prepare_act(task_action), override

    def act(self, key, obs, deterministic=False):
        action, _, _ = self.act_override(key, obs, deterministic=deterministic)
        return action

    def _act(self, key, obs, Q, deterministic, boltzmann_temp, epsilon):
        if deterministic:
            return self.select_action(Q[obs])
        else:
            return self.sample_action(
                key, jnp.asarray(Q[obs], dtype=jnp.float32), boltzmann_temp, epsilon, exploration=self.exploration
            )


