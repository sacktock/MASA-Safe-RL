"""
Base constraint interfaces and Gymnasium wrappers.

This module defines:

- :data:`CostFn`: a callable that maps a set/iterable of atomic proposition labels
  (strings) to a scalar cost.
- :class:`Constraint`: a protocol for stateful constraint monitors that can be
  reset and updated from labels at each environment step.
- :class:`BaseConstraintEnv`: a Gymnasium wrapper that enforces the convention
  that the wrapped environment is a :class:`~masa.common.labelled_env.LabelledEnv`
  and that the constraint monitor is updated using ``info["labels"]``.

The overall convention used throughout MASA is:

1. The base environment (or a wrapper) provides a *labelling function* that maps
   an observation/state to a set of atomic propositions ``labels``.
2. Each call to :meth:`gymnasium.Env.step` returns these labels in the ``info``
   dict under the key ``"labels"``.
3. Constraint monitors are updated as ``constraint.update(labels)``.
4. Constraint wrappers expose metrics for logging/training.

Mathematically, a (labelled) MDP is typically written as

.. math::

   \\mathcal{M} = (\\mathcal{S}, \\mathcal{A}, P, r, L),

where:

- :math:`\\mathcal{S}` is the state space,
- :math:`\\mathcal{A}` is the action space,
- :math:`P(s'\\mid s,a)` is the transition kernel,
- :math:`r(s,a,s')` is a reward signal,
- :math:`L : \\mathcal{S} \\to 2^{\\mathsf{AP}}` is a labelling function mapping
  states to sets of atomic propositions from a finite alphabet :math:`\\mathsf{AP}`.

A cost function then maps labels to a scalar:

.. math::

   c(s) \\triangleq \\mathrm{cost}(L(s)) \\in \\mathbb{R}.

"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Mapping, Protocol, Callable
from masa.common.labelled_env import LabelledEnv
from typing import Dict, Protocol, Any
import gymnasium as gym

CostFn = Callable[Iterable[str], float]

class Constraint(Protocol):
    """Protocol for stateful constraint monitors.

    A :class:`Constraint` is a *monitor* that consumes atomic proposition labels
    at each step and maintains internal state (e.g., cumulative cost, whether an
    LTL automaton is in an accepting/unsafe state, etc.).

    Implementations are intended to be lightweight and compatible with
    Gymnasium wrappers: call :meth:`reset` at episode start and :meth:`update`
    after each environment transition using the label set from ``info["labels"]``.

    Required interface
    ~~~~~~~~~~~~~~~~~~
    Implementations should provide:

    - :meth:`reset`: clear any episode state.
    - :meth:`update`: incorporate the current label set.
    - :attr:`constraint_type`: a stable identifier string for logging/dispatch.

    Metrics interface
    ~~~~~~~~~~~~~~~~~
    The protocol declares:

    - :meth:`step_metric`
    - :meth:`episode_metric`

    """

    def reset(self):
        """Reset any episode-dependent internal state."""

    def update(self, labels: Iterable[str]):
        """Update internal state given the current set of labels.

        Args:
            labels: Iterable of atomic proposition strings active at the current
                step (typically taken from ``info["labels"]``).
        """

    @property
    def constraint_type(self) -> str:
        """A stable identifier for the constraint (e.g., ``"cmdp"``, ``"ltl_safety"``)."""

    
    def step_metric(self) -> Dict[str, float]:
        """Return per-step logging metrics.

        Metrics returned here should be:

        - cheap to compute,
        - non-destructive (do not mutate state),
        - meaningful at *any* time step.

        Examples include running cumulative cost, a per-step violation flag,
        a current probability estimate, etc.

        Returns:
            Dictionary of scalar metrics (values should be JSON/log friendly).
        """

    def episode_metric(self) -> Dict[str, float]:
        """Return end-of-episode logging metrics.

        This is intended to summarize what matters for evaluation/logging at
        episode termination (terminated or truncated).

        Returns:
            Dictionary of scalar metrics (values should be JSON/log friendly).
        """

class BaseConstraintEnv(gym.Wrapper, Constraint):
    """Common base wrapper for constraint-aware environments.

    This wrapper enforces the MASA convention that the wrapped environment is a
    :class:`~masa.common.labelled_env.LabelledEnv` and provides ``info["labels"]``
    as a ``set`` (or ``frozenset``) of atomic propositions at each step.

    The wrapper:

    1. Delegates reset/step to the underlying environment.
    2. Extracts ``labels = info.get("labels", set())``.
    3. Validates that ``labels`` is a set-like container of strings.
    4. Calls ``self._constraint.update(labels)``.

    Attributes:
        env: The wrapped Gymnasium environment (must be a :class:`LabelledEnv`).
        _constraint: The underlying constraint monitor.

    Raises:
        TypeError: If ``env`` is not an instance of :class:`LabelledEnv`.
        ValueError: If ``info["labels"]`` exists but is not a ``set``/``frozenset``.

    Notes:
        The properties :attr:`label_fn` and :attr:`cost_fn` are convenience accessors
        for downstream algorithms. Depending on how wrappers are composed, these
        may be ``None``.
    """

    def __init__(self, env: gym.Env, constraint: Constraint, **kw):
        """Initialize the wrapper.

        Args:
            env: Base environment. Must already be wrapped as a
                :class:`~masa.common.labelled_env.LabelledEnv` so that step/reset
                provide label sets in ``info["labels"]``.
            constraint: A constraint monitor implementing :class:`Constraint`.
            **kw: Unused extra keyword arguments (kept for wrapper compatibility).

        Raises:
            TypeError: If ``env`` is not a :class:`LabelledEnv`.
        """
        if not isinstance(env, LabelledEnv):
            raise TypeError(
                f"{self.__class__.__name__} must wrap a LabelledEnv, "
                f"but got {type(env).__name__}. "
                "Please wrap your environment with LabelledEnv before applying a constraint wrapper."
            )

        super().__init__(env)
        self._constraint = constraint

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """Reset environment and constraint state.

        This calls ``env.reset(...)`` and then resets and updates the constraint
        using the initial label set in ``info["labels"]``.

        Args:
            seed: Optional RNG seed forwarded to the base environment.
            options: Optional reset options forwarded to the base environment.

        Returns:
            A tuple ``(obs, info)`` following the Gymnasium API.

        Raises:
            ValueError: If ``info["labels"]`` is present but not a set/frozenset.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._constraint.reset()

        labels = info.get("labels", set())
        if not isinstance(labels, (set, frozenset)):
            raise ValueError(
                f"Expected 'labels' in info to be a set of atomic propositions, got {type(labels).__name__}"
            )

        self._constraint.update(labels)
        return obs, info

    def step(self, action: Any):
        """Step environment and update constraint from labels.

        Args:
            action: Action to pass to the underlying environment.

        Returns:
            A 5-tuple ``(obs, reward, terminated, truncated, info)`` following
            the Gymnasium API.

        Raises:
            ValueError: If ``info["labels"]`` is present but not a set/frozenset.
        """

        obs, reward, terminated, truncated, info = self.env.step(action)

        labels = info.get("labels", set())
        if not isinstance(labels, (set, frozenset)):
            raise ValueError(
                f"Expected 'labels' in info to be a set of atomic propositions, got {type(labels).__name__}"
            )
            
        self._constraint.update(labels)
        return obs, reward, terminated, truncated, info

    @property
    def cost_fn(self):
        """Expose the cost function if available.

        Returns:
            The underlying cost function if present on the wrapped stack, else
            ``None``.
        """
        if self._constraint is not None:
            return getattr(self.env._constraint, "cost_fn", None)
        return None


    @property
    def label_fn(self):
        """Expose the labelling function if available.

        Returns:
            The environment labelling function if present, else ``None``.
        """
        return getattr(self.env, "label_fn", None)

    @property
    def constraint_type(self) -> str:
        """Constraint identifier forwarded from the underlying monitor."""
        return self._constraint.constraint_type

    def constraint_step_metrics(self) -> Dict[str, float]:
        """Return per-step metrics from the underlying constraint.

        Returns:
            Dictionary of scalar metrics.
        """
        return self._constraint.step_metric()

    def constraint_episode_metrics(self) -> Dict[str, float]:
        """Return end-of-episode metrics from the underlying constraint.

        Returns:
            Dictionary of scalar metrics.
        """
        return self._constraint.episode_metric()