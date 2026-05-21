import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from dataclasses import dataclass
from masa.common.base_class import BaseJaxPolicy
from flax.training.train_state import TrainState
import optax
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import flax.linen as nn
from gymnasium import spaces
from masa.common.layers import Identity, NatureCNN, Flatten
from masa.common.policies import Critic
from typing import Callable, Optional, Sequence, Union, Tuple, Any

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

class ParameterizedActor(nn.Module):
    n_actions: int
    max_successors: int
    trunk_arch: Sequence[int]
    head_arch: Sequence[int]
    conditional_beta_network: bool = True
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    embed_dim: int = 64
    log_std_init: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    eps: float = 1e-6
    mean_clip: Optional[float] = 10.0
    smooth_mean_clip: bool = True

    def _beta_dist(self, loc, scale):
        eps = jnp.array(self.eps, dtype=loc.dtype)
        base = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        bij = tfb.Chain([tfb.Shift(eps), tfb.Scale(1.0 - 2.0 * eps), tfb.Sigmoid()])
        return tfd.TransformedDistribution(distribution=base, bijector=bij)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        return self._forward_features(x)

    def _forward_features(self, x: jnp.ndarray):
        x = Flatten()(x)
        for h in self.trunk_arch:
            x = nn.Dense(h)(x)
            x = self.activation_fn(x)

        logits_i = nn.Dense(self.n_actions)(x)
        logits_j = nn.Dense(self.n_actions)(x)

        return x, logits_i, logits_j

    def _forward_embedding(self):
        emb_i_tbl = self.param("emb_i", nn.initializers.normal(0.02), (self.n_actions, self.embed_dim))
        emb_j_tbl = self.param("emb_j", nn.initializers.normal(0.02), (self.n_actions, self.embed_dim))
        return emb_i_tbl, emb_j_tbl

    def _foward_beta_net(self, x: jnp.ndarray):
        if self.conditional_beta_network:
            for h in self.head_arch:
                x = nn.Dense(h)(x)
                x = self.activation_fn(x)

        loc = nn.Dense(self.max_successors)(x)
        if self.mean_clip is not None:
            if self.smooth_mean_clip:
                loc = self.mean_clip * jnp.tanh(loc)
            else:
                loc = jnp.clip(loc, -self.mean_clip, self.mean_clip)

        log_scale = nn.Dense(self.max_successors)(x) + self.log_std_init
        log_scale = jnp.clip(log_scale, self.log_std_min, self.log_std_max)
        scale = jnp.exp(log_scale) + self.eps

        return loc, scale

    @nn.compact
    def evaluate_actions(self, x: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray,jnp.ndarray]:
        i = actions[:, 0].astype(jnp.int32)
        j = actions[:, 1].astype(jnp.int32)

        y, logits_i, logits_j = self._forward_features(x)

        di = tfd.Categorical(logits=logits_i)
        dj = tfd.Categorical(logits=logits_j)

        if self.conditional_beta_network:
            emb_i_tbl, emb_j_tbl = self._forward_embedding()
            ei = emb_i_tbl[i]
            ej = emb_j_tbl[j]
            y = jnp.concatenate([x, ei, ej], axis=1)

        loc, scale = self._foward_beta_net(y)
        
        beta_dist = self._beta_dist(loc, scale)
        betas = actions[:, 2:].astype(dtype=loc.dtype)

        margin = jnp.array(self.eps*2, dtype=loc.dtype)
        betas = jnp.clip(betas, margin, 1.0 - margin)
        beta_log_prob = beta_dist.log_prob(betas)

        log_prob = di.log_prob(i) + dj.log_prob(j) + beta_log_prob
        entropy = di.entropy() + dj.entropy() - beta_log_prob

        return log_prob, entropy

    @nn.compact
    def sample_action(self, x: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        key_i, key_j, key_b = jax.random.split(key, 3)

        y, logits_i, logits_j = self._forward_features(x)

        di = tfd.Categorical(logits=logits_i)
        dj = tfd.Categorical(logits=logits_j)

        i = di.sample(seed=key_i)
        j = dj.sample(seed=key_j)

        if self.conditional_beta_network:
            emb_i_tbl, emb_j_tbl = self._forward_embedding()
            ei = emb_i_tbl[i]
            ej = emb_j_tbl[j]
            y = jnp.concatenate([x, ei, ej], axis=1)

        loc, scale = self._foward_beta_net(y)

        beta_dist = self._beta_dist(loc, scale)
        betas = beta_dist.sample(seed=key_b)

        margin = jnp.array(self.eps*2, dtype=loc.dtype)
        betas = jnp.clip(betas, margin, 1.0 - margin) # guard against nans
        beta_log_prob = beta_dist.log_prob(betas)

        action = jnp.concatenate([i[:, None], j[:, None], betas], axis=1)
        log_prob = di.log_prob(i) + dj.log_prob(j) + beta_log_prob
        entropy = di.entropy() + dj.entropy() - beta_log_prob

        return action, log_prob, entropy

    @nn.compact
    def select_action(self, x: jnp.ndarray) -> jnp.ndarray:

        y, logits_i, logits_j = self._forward_features(x)
        emb_i_tbl, emb_j_tbl = self._forward_embedding()

        i = jnp.argmax(logits_i, axis=1)
        j = jnp.argmax(logits_j, axis=1)

        if self.conditional_beta_network:
            ei = emb_i_tbl[i]
            ej = emb_j_tbl[j]
            y = jnp.concatenate([x, ei, ej], axis=1)

        loc, _ = self._foward_beta_net(y)

        margin = jnp.array(self.eps*2, dtype=loc.dtype)
        betas = jnp.clip(jax.nn.sigmoid(loc), margin, 1.0 - margin)
        return jnp.concatenate([i[:, None], j[:, None], betas], axis=1)

class ParameterizedPPOPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_actions: int,
        max_successors: int,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        log_std_init: float = -1.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh,
        conditional_beta_network: bool = True,
        features_extractor_class: Optional[type[NatureCNN]] = None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        actor_class: type[nn.Module] = ParameterizedActor,
        critic_class: type[nn.Module] = Critic,
        
    ):
        super().__init__(observation_space, action_space)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == optax.adam:
                optimizer_kwargs["eps"] = 1e-5

        if net_arch is not None:
            if isinstance(net_arch, list):
                self.actor_net_arch = self.critic_net_arch = net_arch
            else:
                assert isinstance(net_arch, dict)
                self.actor_net_arch = net_arch["actor"]
                self.critic_net_arch = net_arch["critic"]
        else:
            self.actor_net_arch = self.critic_net_arch = [64, 64]

        if features_extractor_class is not None:
            if features_extractor_kwargs is None:
                features_extractor_kwargs = {}

        self.n_actions = n_actions
        self.max_successors = max_successors
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        self.conditional_beta_network = conditional_beta_network
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.actor_class = actor_class
        self.critic_class = critic_class

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Union[optax.Schedule, float], max_grad_norm: float) -> jax.Array:
        key, feat_key, actor_key, actor_noise_key, critic_key = jax.random.split(key, 5)
        key, self.key = jax.random.split(key, 2)

        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        if self.features_extractor_class is not None:
            self.featurizer = self.features_extractor_class(
                **self.features_extractor_kwargs
            )
        else:
            self.featurizer = Identity()

        optimizer = self.optimizer_class(
            learning_rate=lr_schedule,
            **self.optimizer_kwargs,
        )

        self.featurizer_state = TrainState.create(
            apply_fn=self.featurizer.apply,
            params=self.featurizer.init(feat_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.featurizer.apply = jit(self.featurizer.apply)

        obs = self.featurizer.apply(self.featurizer_state.params, obs)

        self.actor = self.actor_class(
            n_actions=self.n_actions,
            max_successors=self.max_successors,
            trunk_arch=self.actor_net_arch,
            head_arch=self.actor_net_arch,
            conditional_beta_network=self.conditional_beta_network,
            activation_fn=self.activation_fn,
            log_std_init=self.log_std_init
        )

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs, actor_noise_key, method="sample_action"),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.critic = self.critic_class(
            net_arch=self.critic_net_arch, 
            activation_fn=self.activation_fn
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.actor.apply = jit(self.actor.apply)
        self.critic.apply = jit(self.critic.apply)

        return key

    def reset_noise(self):
        self.key, self.noise_key = jax.random.split(self.key)
 
    def forward(self, key: jax.Array, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(key, obs, deterministic=deterministic)

    def predict_all(self, key: jax.Array, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_all(key, self.featurizer_state, self.actor_state, self.critic_state, observation)

    def _predict(self, key: jax.Array, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return self.select_action(self.featurizer_state, self.actor_state, obs)
        else:
            return self.sample_action(key, self.featurizer_state, self.actor_state, obs)

    @staticmethod
    @jit
    def _predict_all(
        key: jax.Array, featurizer_state: TrainState, actor_state: TrainState, critic_state: TrainState, observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        feats = featurizer_state.apply_fn(featurizer_state.params, observations)
        actions, log_probs, entropy = actor_state.apply_fn(
            actor_state.params,
            feats,
            key,
            method="sample_action",
        )
        values = critic_state.apply_fn(critic_state.params, feats).flatten()
        return actions, log_probs, values

    @staticmethod
    @jit
    def select_action(
        featurizer_state: TrainState, actor_state: TrainState, obs: jnp.ndarray
    ) -> jnp.ndarray:
        feats = featurizer_state.apply_fn(featurizer_state.params, obs)
        action, _, _ = actor_state.apply_fn(
            actor_state.params,
            feats,
            method="select_action",
        )
        return action
        
    @staticmethod
    @jit
    def sample_action(
        key: jax.Array, featurizer_state: TrainState, actor_state: TrainState, obs: jnp.ndarray
    ) -> jnp.ndarray:
        feats = featurizer_state.apply_fn(featurizer_state.params, obs)
        action, _, _ = actor_state.apply_fn(
            actor_state.params,
            feats,
            key,
            method="sample_action",
        )
        return action
