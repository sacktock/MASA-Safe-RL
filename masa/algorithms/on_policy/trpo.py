from __future__ import annotations
import jax.random as jr
import jax.numpy as jnp
import optax
from jax import jit
import jax
from functools import partial
from flax.training.train_state import TrainState
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional, TypeVar, Union, Callable
from masa.common.base_class import BaseJaxPolicy
from masa.common.on_policy_algorithm import OnPolicyAlgorithm
from masa.common.policies import PPOPolicy
from tqdm.auto import tqdm

class TRPO(OnPolicyAlgorithm):

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
        learning_rate: Union[float, optax.Schedule] = 3e-4,
        n_steps: int = 2048,
        n_critic_updates: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        fvp_sample_freq: int = 4,
        line_search_steps: int = 10,
        line_search_decay: float = 0.8,
        policy_class: type[BaseJaxPolicy] = PPOPolicy,
        policy_kwargs: Optional[dict[str, Any]] = None,
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
            use_tqdm_rollout=True, # Turn on tqdm progress bar for rollout
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs
        )

        if normalize_advantage:
            assert n_steps > 1, "n_steps must be > 1 when normalize_advantage = True"

        self.normalize_advantage = normalize_advantage
        self.n_critic_updates = n_critic_updates
        self.target_kl = target_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.fvp_sample_freq = fvp_sample_freq
        self.line_search_steps = line_search_steps
        self.line_search_decay = line_search_decay

    @staticmethod
    @jit
    def flatten_params(params):
        leaves, _ = jax.tree_util.tree_flatten(params)
        return jnp.concatenate([jnp.ravel(x) for x in leaves])

    @staticmethod
    def unflatten_params(flat_params, reference_tree):
        leaves, treedef = jax.tree_util.tree_flatten(reference_tree)
        new_leaves = []
        idx = 0
        for leaf in leaves:
            size = leaf.size
            new_leaf = flat_params[idx: idx + size].reshape(leaf.shape)
            new_leaves.append(new_leaf)
            idx += size
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    @staticmethod
    def tree_add(tree_a, tree_b, alpha=1.0):
        return jax.tree.map(lambda a, b: a + alpha * b, tree_a, tree_b)

    @staticmethod
    @jit
    def tree_dot(tree_a, tree_b):
        leaves_a, _ = jax.tree_util.tree_flatten(tree_a)
        leaves_b, _ = jax.tree_util.tree_flatten(tree_b)
        return sum([jnp.sum(a * b) for a, b in zip(leaves_a, leaves_b)])
        
    @staticmethod
    def conjugate_gradient(Ax, b, iters=10):
        x = jax.tree.map(jnp.zeros_like, b)
        r = jax.tree.map(lambda x: x.copy(), b)
        p = jax.tree.map(lambda x: x.copy(), r)
        rs_old = TRPO.tree_dot(r, r)
        for _ in range(iters):
            Ap = Ax(p)
            alpha = rs_old / (TRPO.tree_dot(p, Ap) + 1e-8)
            x = jax.tree.map(lambda x_, p_: x_ + alpha * p_, x, p)
            r = jax.tree.map(lambda r_, Ap_: r_ - alpha * Ap_, r, Ap)
            rs_new = TRPO.tree_dot(r, r)
            beta = rs_new / (rs_old + 1e-8)
            p = jax.tree.map(lambda r_, p_: r_ + beta * p_, r, p)
            rs_old = rs_new
        return x

    @staticmethod
    @jit
    def _value_update(
        featurizer_state: TrainState,
        critic_state: TrainState,
        observations: jnp.ndarray,
        returns: jnp.ndarray,
        vf_coef: float,
    ):
        def critic_loss(featurizer_params, critic_params):
            features = featurizer_state.apply_fn(featurizer_params, observations)
            # Critic loss
            critic_values = critic_state.apply_fn(critic_params, features).flatten()
            value_loss = vf_coef * ((returns - critic_values)**2).mean()
            return value_loss

        loss, grads = jax.value_and_grad(critic_loss, argnums=(0, 1))(
            featurizer_state.params, critic_state.params
        )

        featurizer_state = featurizer_state.apply_gradients(grads=grads[0])
        critic_state = critic_state.apply_gradients(grads=grads[1])

        return (featurizer_state, critic_state), loss

    @staticmethod
    @jit
    def _surrogate_loss(
        actor_params: Any,
        featurizer_state: TrainState,
        actor_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        advantages: jnp.ndarray,
        old_log_prob: jnp.ndarray,
        ent_coef: float,
    ):
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        dist = actor_state.apply_fn(actor_params, features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        ratio = jnp.exp(log_prob - old_log_prob)
        policy_loss = -(ratio * advantages).mean()

        entropy_loss = -jnp.mean(entropy)
        return policy_loss + ent_coef * entropy_loss

    @staticmethod
    @jit
    def _mean_kl(
        old_actor_params: Any,
        new_actor_params: Any,
        featurizer_state: TrainState,
        actor_state: TrainState,
        observations: jnp.ndarray,
    ):
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        old_dist = actor_state.apply_fn(old_actor_params, features)
        new_dist = actor_state.apply_fn(new_actor_params, features)

        kl = old_dist.kl_divergence(new_dist)
        return kl.mean()

    @staticmethod
    def _fisher_vector_product(
        actor_params: Any,
        vector: jnp.ndarray,
        featurizer_state: TrainState,
        actor_state: TrainState,
        observations: jnp.ndarray,
        cg_damping: float,
    ):
        def kl_fn(params):
            return TRPO._mean_kl(actor_params, params, featurizer_state, actor_state, observations)
        grad_fn = jax.grad(kl_fn)
        vector_tree = TRPO.unflatten_params(vector, actor_params)
        _, hvp = jax.jvp(grad_fn, (actor_params,), (vector_tree,))
        flat_hvp = TRPO.flatten_params(hvp)
        return flat_hvp + cg_damping * vector

    def optimize(
        self,
        step: int, 
        logger: Optional[TrainLogger] = None,
        tqdm_position: int = 1
    ):

        self.key, subkey = jr.split(self.key)
        rollout_data = next(self.rollout_buffer.get(subkey, self.n_steps))
        observations, actions, rewards, values, returns, advantages, old_log_probs = rollout_data

        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to int
            actions = actions.flatten().astype(np.int32)

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_actor_params = self.policy.actor_state.params

        def actor_loss(actor_params):
            return self._surrogate_loss(
                actor_params,
                self.policy.featurizer_state,
                self.policy.actor_state,
                observations,
                actions,
                advantages,
                old_log_probs,
                self.ent_coef,
            )

        loss_before, grads = jax.value_and_grad(actor_loss)(old_actor_params)

        flat_grads = self.flatten_params(grads)

        fvp_observations = observations[::self.fvp_sample_freq]

        def fvp(v):
            return self._fisher_vector_product(
                old_actor_params,
                v,
                self.policy.featurizer_state,
                self.policy.actor_state,
                fvp_observations,
                self.cg_damping,
            )

        step_direction = self.conjugate_gradient(
            fvp,
            -flat_grads,
            self.cg_iters,
        )

        shs = 0.5 * jnp.dot(
            step_direction,
            fvp(step_direction),
        )

        step_size = jnp.sqrt(
            self.target_kl / (shs + 1e-8)
        )

        full_step = step_direction * step_size

        old_flat = self.flatten_params(old_actor_params)

        expected_improve = -jnp.dot(flat_grads, full_step)

        accepted = False

        for i in range(self.line_search_steps):
            frac = self.line_search_decay ** i
            candidate = old_flat + frac * full_step
            candidate_params = self.unflatten_params(
                candidate,
                old_actor_params,
            )
            loss = actor_loss(candidate_params)
            kl = self._mean_kl(
                old_actor_params, 
                candidate_params, 
                self.policy.featurizer_state,
                self.policy.actor_state,
                observations,
            )
            improve = loss_before - loss
            last_kl = kl
            if improve > 0 and kl < self.target_kl:
                self.policy.actor_state = self.policy.actor_state.replace(params=candidate_params)
                accepted = True
                break

        with tqdm(
                total=self.n_critic_updates,
                desc="optimize_critic",
                position=tqdm_position,
                leave=False,
                dynamic_ncols=True,
                colour="cyan",
        ) as pbar:

            for _ in range(self.n_critic_updates):
                (self.policy.featurizer_state, self.policy.critic_state), vf_loss = \
                self._value_update(
                    featurizer_state=self.policy.featurizer_state,
                    critic_state=self.policy.critic_state,
                    observations=observations,
                    returns=returns,
                    vf_coef=self.vf_coef,
                )

                pbar.update(1)

        if logger:
            logger.add(
                "train/stats",{
                "policy_loss": float(loss_before),
                "value_loss": float(vf_loss),
                "kl": float(kl),
                "accepted": float(accepted),
                "expected_improve": float(expected_improve),
                "step_size": float(step_size),
                "grad_norm": float(jnp.linalg.norm(flat_grads)),
                "xHx": float(shs * 2.0),
                "final_kl": float(last_kl),
                "lr": float(self.lr_schedule(step)),
            })

    @property
    def train_ratio(self):
        return self.n_steps * self.n_envs