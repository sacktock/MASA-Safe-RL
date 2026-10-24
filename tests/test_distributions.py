"""Regression tests for the JAX-native distributions in ``masa.common``.

These lock in the analytic ``mean``/``mode``/``entropy``/``kl_divergence``
behaviour that replaced ``tensorflow_probability`` (and in particular guard the
CPO/TRPO ``kl_divergence`` path). Reference values are computed independently
with NumPy so the tests fail if the closed forms drift.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from masa.common import distributions as dists

_LOG2PI = np.log(2.0 * np.pi)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def _cat_entropy_np(logits: np.ndarray) -> np.ndarray:
    p = _softmax(logits)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)


def _cat_kl_np(logits_p: np.ndarray, logits_q: np.ndarray) -> np.ndarray:
    p = _softmax(logits_p)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(p)
        log_q = np.log(_softmax(logits_q))
        return np.sum(np.where(p > 0, p * (log_p - log_q), 0.0), axis=-1)


def _mvn_entropy_np(scale: np.ndarray) -> np.ndarray:
    return np.sum(0.5 * (_LOG2PI + 1.0) + np.log(scale), axis=-1)


def _mvn_kl_np(mp, sp, mq, sq) -> np.ndarray:
    return np.sum(
        np.log(sq) - np.log(sp) + (sp**2 + (mp - mq) ** 2) / (2.0 * sq**2) - 0.5,
        axis=-1,
    )


# --------------------------------------------------------------------------- #
# Categorical
# --------------------------------------------------------------------------- #
CAT_LOGITS_P = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 0.5]])
CAT_LOGITS_Q = np.array([[0.2, 1.0, 1.5], [1.0, 0.0, -0.5]])


def test_categorical_entropy_matches_numpy():
    dist = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P))
    np.testing.assert_allclose(
        np.asarray(dist.entropy()), _cat_entropy_np(CAT_LOGITS_P), rtol=1e-6
    )


def test_categorical_mode_is_argmax():
    dist = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P))
    np.testing.assert_array_equal(
        np.asarray(dist.mode()), np.argmax(CAT_LOGITS_P, axis=-1)
    )


def test_categorical_kl_matches_numpy_and_self_kl_is_zero():
    p = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P))
    q = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_Q))
    np.testing.assert_allclose(
        np.asarray(p.kl_divergence(q)),
        _cat_kl_np(CAT_LOGITS_P, CAT_LOGITS_Q),
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(p.kl_divergence(p)), 0.0, atol=1e-6)


def test_categorical_kl_handles_minus_inf_padding_without_nan():
    # MultiDiscrete actors pad invalid choices with -inf logits.
    p = dists.Categorical(logits=jnp.array([1.0, 2.0, -jnp.inf]))
    q = dists.Categorical(logits=jnp.array([0.5, 1.0, -jnp.inf]))
    kl = np.asarray(p.kl_divergence(q))
    assert np.isfinite(kl).all()
    np.testing.assert_allclose(
        kl,
        _cat_kl_np(np.array([1.0, 2.0, -np.inf]), np.array([0.5, 1.0, -np.inf])),
        rtol=1e-5,
    )


def test_categorical_mean_raises():
    dist = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P))
    with pytest.raises(NotImplementedError):
        dist.mean()


# --------------------------------------------------------------------------- #
# MultivariateNormalDiag
# --------------------------------------------------------------------------- #
MVN_LOC_P = np.array([[0.0, 1.0], [-1.0, 2.0]])
MVN_SCALE_P = np.array([[1.0, 2.0], [0.5, 1.5]])
MVN_LOC_Q = np.array([[0.5, 0.0], [-1.5, 1.0]])
MVN_SCALE_Q = np.array([[1.5, 1.0], [1.0, 2.0]])


def _mvn_p():
    return dists.MultivariateNormalDiag(
        loc=jnp.asarray(MVN_LOC_P), scale_diag=jnp.asarray(MVN_SCALE_P)
    )


def test_mvn_mean_and_mode_are_loc():
    dist = _mvn_p()
    np.testing.assert_allclose(np.asarray(dist.mean()), MVN_LOC_P)
    np.testing.assert_allclose(np.asarray(dist.mode()), MVN_LOC_P)


def test_mvn_entropy_matches_numpy():
    dist = _mvn_p()
    np.testing.assert_allclose(
        np.asarray(dist.entropy()), _mvn_entropy_np(MVN_SCALE_P), rtol=1e-6
    )


def test_mvn_kl_matches_numpy_and_self_kl_is_zero():
    p = _mvn_p()
    q = dists.MultivariateNormalDiag(
        loc=jnp.asarray(MVN_LOC_Q), scale_diag=jnp.asarray(MVN_SCALE_Q)
    )
    np.testing.assert_allclose(
        np.asarray(p.kl_divergence(q)),
        _mvn_kl_np(MVN_LOC_P, MVN_SCALE_P, MVN_LOC_Q, MVN_SCALE_Q),
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(p.kl_divergence(p)), 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Independent
# --------------------------------------------------------------------------- #
def test_independent_reduces_entropy_and_kl_over_batch():
    base_p = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P))
    base_q = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_Q))
    ind_p = dists.Independent(base_p, reinterpreted_batch_ndims=1)
    ind_q = dists.Independent(base_q, reinterpreted_batch_ndims=1)

    np.testing.assert_allclose(
        np.asarray(ind_p.entropy()), _cat_entropy_np(CAT_LOGITS_P).sum(), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(ind_p.kl_divergence(ind_q)),
        _cat_kl_np(CAT_LOGITS_P, CAT_LOGITS_Q).sum(),
        rtol=1e-6,
    )


def test_independent_mode_delegates():
    base = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P))
    ind = dists.Independent(base, reinterpreted_batch_ndims=1)
    np.testing.assert_array_equal(
        np.asarray(ind.mode()), np.argmax(CAT_LOGITS_P, axis=-1)
    )


def test_independent_mean_delegates_to_base():
    # Wrapping an MVN: mean is well-defined and delegated.
    mvn = _mvn_p()
    ind = dists.Independent(mvn, reinterpreted_batch_ndims=1)
    np.testing.assert_allclose(np.asarray(ind.mean()), MVN_LOC_P)

    # Wrapping a Categorical: mean is undefined and must propagate the error.
    cat = dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P))
    with pytest.raises(NotImplementedError):
        dists.Independent(cat, reinterpreted_batch_ndims=1).mean()


# --------------------------------------------------------------------------- #
# TransformedDistribution
# --------------------------------------------------------------------------- #
def _transformed(loc, scale):
    base = dists.MultivariateNormalDiag(
        loc=jnp.asarray(loc), scale_diag=jnp.asarray(scale)
    )
    return base, dists.TransformedDistribution(base, dists.Sigmoid())


def test_transformed_mode_is_pushforward_of_base_mode():
    loc = np.array([[0.0, 0.5], [-1.0, 2.0]])
    scale = np.ones_like(loc)
    _, td = _transformed(loc, scale)
    np.testing.assert_allclose(
        np.asarray(td.mode()), np.asarray(jax.nn.sigmoid(jnp.asarray(loc))), rtol=1e-6
    )


def test_transformed_kl_delegates_to_base_for_shared_bijector():
    base_p, td_p = _transformed(MVN_LOC_P, MVN_SCALE_P)
    base_q, td_q = _transformed(MVN_LOC_Q, MVN_SCALE_Q)
    np.testing.assert_allclose(
        np.asarray(td_p.kl_divergence(td_q)),
        np.asarray(base_p.kl_divergence(base_q)),
        rtol=1e-6,
    )


def test_transformed_mean_and_entropy_raise():
    _, td = _transformed(MVN_LOC_P, MVN_SCALE_P)
    with pytest.raises(NotImplementedError):
        td.mean()
    with pytest.raises(NotImplementedError):
        td.entropy()


# --------------------------------------------------------------------------- #
# API surface: every distribution exposes the four methods.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dist",
    [
        dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P)),
        dists.MultivariateNormalDiag(
            loc=jnp.asarray(MVN_LOC_P), scale_diag=jnp.asarray(MVN_SCALE_P)
        ),
        dists.Independent(
            dists.Categorical(logits=jnp.asarray(CAT_LOGITS_P)),
            reinterpreted_batch_ndims=1,
        ),
        dists.TransformedDistribution(
            dists.MultivariateNormalDiag(
                loc=jnp.asarray(MVN_LOC_P), scale_diag=jnp.asarray(MVN_SCALE_P)
            ),
            dists.Sigmoid(),
        ),
    ],
)
@pytest.mark.parametrize("method", ["mean", "mode", "entropy", "kl_divergence"])
def test_methods_are_defined(dist, method):
    assert callable(getattr(dist, method))
