
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
import jax
import jax.random as jr

class RolloutBuffer():

    def __init__(
        self, 
        buffer_size: int, 
        observation_space: spaces.Space, 
        action_space: spaces.Space, 
        n_envs: int, 
        gamma: float = 0.99, 
        gae_lambda: float = 1.0,
    ):

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.obs_shape = self.observation_space.shape

        self.act_dim = self._compute_act_dim(self.action_space)

    @staticmethod
    def _compute_act_dim(space: spaces.Space) -> int:
        """
        Compute the flattened action dimension for any supported Gymnasium space.
        For Dict/Tuple spaces, sum the dims of subspaces (iterate over keys / elements).
        """
        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        if isinstance(space, spaces.Discrete):
            return 1
        if isinstance(space, spaces.MultiDiscrete):
            return int(len(space.nvec))
        if isinstance(space, spaces.MultiBinary):
            return int(space.n)

        if isinstance(space, spaces.Dict):
            return int(sum(RolloutBuffer._compute_act_dim(sub) for sub in space.spaces.values()))

        if isinstance(space, spaces.Tuple):
            return int(sum(RolloutBuffer._compute_act_dim(sub) for sub in space.spaces))

        raise NotImplementedError(f"RolloutBuffer does not support action space type {type(space).__name__}")

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    def compute_returns_and_advantages(self, last_value: Optional[np.ndarray] = None, done: np.ndarray = np.array(False)):

        if last_value is None:
            last_value = np.array(0.0)

        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == (self.buffer_size-1):
                next_non_terminal = 1.0 - done.astype(np.float32)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step+1].astype(np.float32)
                next_value = self.values[step+1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def add(
        self, 
        obs: np.ndarray, 
        actions: np.ndarray,
        rewards: np.ndarray, 
        episode_starts: np.ndarray,
        values: np.ndarray, 
        log_probs: np.ndarray
    ):

        if self.full:
            return False

        if len(log_probs.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_probs = log_probs.reshape(-1, 1)

        actions = actions.reshape((self.n_envs, self.act_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(actions)
        self.rewards[self.pos] = np.array(rewards)
        self.episode_starts[self.pos] = np.array(episode_starts)
        self.values[self.pos] = np.array(values)
        self.log_probs[self.pos] = np.array(log_probs)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        
        return True

    def get(self, key: jax.Array, batch_size: int):
        assert self.full
        indices = jr.permutation(key, self.buffer_size)

        if batch_size is None:
            batch_size=self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.observations[batch_inds].reshape(-1, *self.obs_shape),
            self.actions[batch_inds].reshape(-1, self.act_dim),
            self.rewards[batch_inds].flatten(),
            self.values[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten()
        )
        return data

class CostRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with additional storage for constraint costs and
    cost-based returns / advantages.
    """

    def __init__(
        self, 
        buffer_size: int, 
        observation_space: spaces.Space, 
        action_space: spaces.Space, 
        n_envs: int, 
        gamma: float = 0.99, 
        gae_lambda: float = 1.0,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 1.0,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            n_envs,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda

    def reset(self):
        super().reset()

        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def compute_returns_and_advantages(
        self,
        last_value: Optional[np.ndarray] = None,
        last_cost_value: Optional[np.ndarray] = None,
        done: np.ndarray = np.array(False),
    ):

        if last_value is None:
            last_value = np.array(0.0)

        if last_cost_value is None:
            last_cost_value = np.array(0.0)

        last_gae_lam = 0
        last_cost_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == (self.buffer_size-1):
                next_non_terminal = 1.0 - done.astype(np.float32)
                next_value = last_value
                next_cost_value = last_cost_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step+1].astype(np.float32)
                next_value = self.values[step+1]
                next_cost_value = self.cost_values[step+1]
            # reward
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

            # cost
            cost_delta = self.costs[step] + self.cost_gamma * next_cost_value * next_non_terminal - self.cost_values[step]
            last_cost_gae_lam = cost_delta + self.cost_gamma * self.cost_gae_lambda * next_non_terminal * last_cost_gae_lam
            self.cost_advantages[step] = last_cost_gae_lam

        self.returns = self.advantages + self.values
        self.cost_returns = self.cost_advantages + self.cost_values

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        episode_starts: np.ndarray,
        values: np.ndarray,
        cost_values: np.ndarray,
        log_probs: np.ndarray,
    ):

        success = super().add(
            obs=obs,
            actions=actions,
            rewards=rewards,
            episode_starts=episode_starts,
            values=values,
            log_probs=log_probs,
        )

        if not success:
            return False

        self.costs[self.pos-1] = np.array(costs)
        self.cost_values[self.pos-1] = np.array(cost_values)

        return True

    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.observations[batch_inds].reshape(-1, *self.obs_shape),
            self.actions[batch_inds].reshape(-1, self.act_dim),
            self.rewards[batch_inds].flatten(),
            self.costs[batch_inds].flatten(),
            self.values[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
        )
        return data

class VTraceRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with additional storage for the shieled action log probs
    and V-trace returns and advantage estimation.
    """

    def __init__(
        self, 
        buffer_size: int, 
        observation_space: spaces.Space, 
        action_space: spaces.Space, 
        n_envs: int, 
        gamma: float = 0.99, 
        gae_lambda: float = 1.0,
        clip_rho: float = 1.0,
        clip_c: float = 1.0,
        clip_traj: bool = False
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            n_envs,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self.clip_rho = clip_rho # TD-error IS clipping ratio
        self.clip_c = clip_c # recursive trace cutting clipping
        self.clip_traj = clip_traj

    def reset(self):
        super().reset()

        self.shield_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def compute_returns_and_advantages(self, last_value: Optional[np.ndarray] = None, done: np.ndarray = np.array(False)):

        if last_value is None:
            last_value = np.array(0.0)

        last_gae_lam = 0.0

        ratios = np.exp(self.log_probs - self.shield_log_probs)

        deltas = np.zeros_like(ratios)
        acc_ratios = np.ones_like(ratios)

        for step in reversed(range(self.buffer_size)):
            if step == (self.buffer_size-1):
                next_non_terminal = 1.0 - done.astype(np.float32)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step+1].astype(np.float32)
                next_value = self.values[step+1]

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            deltas[step] = delta

            if self.clip_traj:
                # trajectory-level cumulative IS ratio clipping
                rho_traj = np.minimum(ratios[step] * acc_ratios[step:], self.clip_rho)

                decay_weights = (self.gamma * self.gae_lambda) ** (np.arange(step, self.buffer_size) - step)
                self.advantages[step] = np.sum(rho_traj * deltas[step:] * decay_weights[:, np.newaxis], axis=0)

                # accumulate ratios
                acc_ratios[step:] *= ratios[step][np.newaxis, :]
                acc_ratios[step:] *= (next_non_terminal)[np.newaxis, :]
            else:
                rho = np.minimum(ratios[step], self.clip_rho)
                c = np.minimum(ratios[step], self.clip_c)

                last_gae_lam = rho * delta + self.gamma * self.gae_lambda * next_non_terminal * c * last_gae_lam
                self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def add(
        self, 
        obs: np.ndarray, 
        actions: np.ndarray,
        rewards: np.ndarray, 
        episode_starts: np.ndarray,
        values: np.ndarray, 
        log_probs: np.ndarray,
        shield_log_probs: np.ndarray,
    ):

        success = super().add(
            obs=obs,
            actions=actions,
            rewards=rewards,
            episode_starts=episode_starts,
            values=values,
            log_probs=log_probs,
        )

        if not success:
            return False

        self.shield_log_probs[self.pos-1] = np.array(shield_log_probs)

        return True

    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.observations[batch_inds].reshape(-1, *self.obs_shape),
            self.actions[batch_inds].reshape(-1, self.act_dim),
            self.rewards[batch_inds].flatten(),
            self.values[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.shield_log_probs[batch_inds].flatten()
        )
        return data

