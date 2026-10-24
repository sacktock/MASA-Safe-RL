"""Minimal, JAX-native probability distributions and bijectors.

This module provides the small, closed set of distributions and bijectors that
MASA's JAX policies need, replacing ``tensorflow_probability.substrates.jax``.

The implementations below are intentionally small and jit-friendly. They match
the subset of the TFP API used in this codebase:

* :class:`Categorical` (``logits=`` or ``probs=``)
* :class:`MultivariateNormalDiag`
* :class:`Independent`
* :class:`TransformedDistribution` with :class:`Chain` / :class:`Shift` /
  :class:`Scale` / :class:`Sigmoid` bijectors

Each distribution exposes ``sample(seed=...)``, ``log_prob``, ``entropy`` and
``mode`` where the original code requires them, using the same keyword
conventions as TFP so call sites remain unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp

_LOG2PI = jnp.log(2.0 * jnp.pi)


class Distribution:
    """Base class so distributions share a common type for annotations."""


class Categorical(Distribution):
    """Categorical distribution over the last axis of ``logits``/``probs``."""

    def __init__(
        self, logits: Optional[jnp.ndarray] = None, probs: Optional[jnp.ndarray] = None
    ):
        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of `logits` or `probs`.")
        if logits is not None:
            self.log_probs = jax.nn.log_softmax(logits, axis=-1)
        else:
            probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
            # log(0) -> -inf is fine for sampling/log_prob; entropy guards it.
            self.log_probs = jnp.log(probs)

    def sample(self, seed) -> jnp.ndarray:
        return jax.random.categorical(seed, self.log_probs, axis=-1)

    def log_prob(self, value) -> jnp.ndarray:
        value = value.astype(jnp.int32)
        gathered = jnp.take_along_axis(self.log_probs, value[..., None], axis=-1)
        return gathered[..., 0]

    def entropy(self) -> jnp.ndarray:
        probs = jnp.exp(self.log_probs)
        terms = jnp.where(probs > 0, probs * self.log_probs, 0.0)
        return -jnp.sum(terms, axis=-1)

    def mode(self) -> jnp.ndarray:
        return jnp.argmax(self.log_probs, axis=-1)

    def mean(self):
        raise NotImplementedError(
            "Categorical.mean() is not defined; categories are not numeric."
        )

    def kl_divergence(self, other: Categorical) -> jnp.ndarray:
        probs = jnp.exp(self.log_probs)
        # -inf guard
        terms = jnp.where(probs > 0, probs * (self.log_probs - other.log_probs), 0.0)
        return jnp.sum(terms, axis=-1)


class MultivariateNormalDiag(Distribution):
    """Diagonal-covariance multivariate normal (event = last axis)."""

    def __init__(self, loc, scale_diag):
        self.loc = loc
        self.scale_diag = scale_diag

    def sample(self, seed) -> jnp.ndarray:
        noise = jax.random.normal(seed, self.loc.shape, self.loc.dtype)
        return self.loc + self.scale_diag * noise

    def log_prob(self, value) -> jnp.ndarray:
        z = (value - self.loc) / self.scale_diag
        per_dim = -0.5 * (z**2 + _LOG2PI) - jnp.log(self.scale_diag)
        return jnp.sum(per_dim, axis=-1)

    def entropy(self) -> jnp.ndarray:
        per_dim = 0.5 * (_LOG2PI + 1.0) + jnp.log(self.scale_diag)
        return jnp.sum(per_dim, axis=-1)

    def mode(self) -> jnp.ndarray:
        return self.loc

    def mean(self) -> jnp.ndarray:
        return self.loc

    def kl_divergence(self, other: MultivariateNormalDiag) -> jnp.ndarray:
        var_p = self.scale_diag**2
        var_q = other.scale_diag**2
        per_dim = (
            jnp.log(other.scale_diag)
            - jnp.log(self.scale_diag)
            + (var_p + (self.loc - other.loc) ** 2) / (2.0 * var_q)
            - 0.5
        )
        return jnp.sum(per_dim, axis=-1)


class Independent(Distribution):
    """Reinterpret the last ``reinterpreted_batch_ndims`` batch axes as event."""

    def __init__(self, distribution: Distribution, reinterpreted_batch_ndims: int = 1):
        self.distribution = distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def _reduce(self, value) -> jnp.ndarray:
        for _ in range(self.reinterpreted_batch_ndims):
            value = jnp.sum(value, axis=-1)
        return value

    def sample(self, seed) -> jnp.ndarray:
        return self.distribution.sample(seed=seed)

    def log_prob(self, value) -> jnp.ndarray:
        return self._reduce(self.distribution.log_prob(value))

    def entropy(self) -> jnp.ndarray:
        return self._reduce(self.distribution.entropy())

    def mode(self) -> jnp.ndarray:
        return self.distribution.mode()

    def mean(self) -> jnp.ndarray:
        return self.distribution.mean()

    def kl_divergence(self, other: Independent) -> jnp.ndarray:
        return self._reduce(self.distribution.kl_divergence(other.distribution))


# ---------------------------------------------------------------------------
# Bijectors (elementwise). ``forward_log_det_jacobian`` returns the per-element
# log|dforward/dx|; TransformedDistribution sums it over the event axis.
# ---------------------------------------------------------------------------
class Bijector:
    def forward(self, x):  # pragma: no cover - interface
        raise NotImplementedError

    def inverse(self, y):  # pragma: no cover - interface
        raise NotImplementedError

    def forward_log_det_jacobian(self, x):  # pragma: no cover - interface
        raise NotImplementedError


class Sigmoid(Bijector):
    def forward(self, x) -> jnp.ndarray:
        return jax.nn.sigmoid(x)

    def inverse(self, y) -> jnp.ndarray:
        return jax.scipy.special.logit(y)

    def forward_log_det_jacobian(self, x) -> jnp.ndarray:
        # log(sigmoid(x)) + log(1 - sigmoid(x))
        return -jax.nn.softplus(-x) - jax.nn.softplus(x)


class Scale(Bijector):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, x) -> jnp.ndarray:
        return self.scale * x

    def inverse(self, y) -> jnp.ndarray:
        return y / self.scale

    def forward_log_det_jacobian(self, x) -> jnp.ndarray:
        return jnp.broadcast_to(jnp.log(jnp.abs(self.scale)), jnp.shape(x))


class Shift(Bijector):
    def __init__(self, shift):
        self.shift = shift

    def forward(self, x) -> jnp.ndarray:
        return x + self.shift

    def inverse(self, y) -> jnp.ndarray:
        return y - self.shift

    def forward_log_det_jacobian(self, x) -> jnp.ndarray:
        return jnp.zeros_like(x)


class Chain(Bijector):
    """Compose bijectors. ``Chain([f, g, h]).forward(x) == f(g(h(x)))``."""

    def __init__(self, bijectors: Sequence[Bijector]):
        self.bijectors = list(bijectors)

    def forward(self, x) -> jnp.ndarray:
        for bijector in reversed(self.bijectors):
            x = bijector.forward(x)
        return x

    def inverse(self, y) -> jnp.ndarray:
        for bijector in self.bijectors:
            y = bijector.inverse(y)
        return y

    def forward_log_det_jacobian(self, x) -> jnp.ndarray:
        total = jnp.zeros_like(x)
        for bijector in reversed(self.bijectors):
            total = total + bijector.forward_log_det_jacobian(x)
            x = bijector.forward(x)
        return total


class TransformedDistribution(Distribution):
    """Push a base distribution through an elementwise bijector.

    Assumes the base distribution's ``log_prob`` already reduces over its event
    (last) axis and that ``bijector`` acts elementwise, so the log-det-Jacobian
    is summed over the last axis - matching the diagonal-Gaussian usage here.
    """

    def __init__(self, distribution: Distribution, bijector: Bijector):
        self.distribution = distribution
        self.bijector = bijector

    def sample(self, seed) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.sample(seed=seed))

    def log_prob(self, value):
        x = self.bijector.inverse(value)
        fldj = jnp.sum(self.bijector.forward_log_det_jacobian(x), axis=-1)
        return self.distribution.log_prob(x) - fldj

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def mean(self):
        raise NotImplementedError(
            "TransformedDistribution.mean() is not implemented; the mean does "
            "not map through a nonlinear bijector in closed form."
        )

    def entropy(self):
        raise NotImplementedError(
            "TransformedDistribution.entropy() is not implemented; the "
            "log-det-Jacobian correction has no closed form in general."
        )

    def kl_divergence(self, other: TransformedDistribution) -> jnp.ndarray:
        return self.distribution.kl_divergence(other.distribution)
