from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
from masa.common.constraints.base import BaseConstraintEnv
from masa.common.ltl import DFACostFn, DFA, ShapedCostFn
from masa.common.running_mean_std import RunningMeanStd
import numpy as np
from collections import deque
from tqdm import tqdm

def is_wrapped(env: gym.Env, wrapper_class: gym.Wrapper) -> bool:
    r"""
    Check whether ``env`` is wrapped (anywhere in its wrapper chain) by
    ``wrapper_class``.

    This helper walks through typical wrapper chains:

    * Gymnasium-style wrappers via ``.env``.
    * Vector-env style wrappers via a ``.venv`` attribute (commonly used by
      vectorized environments and some third-party libraries).

    Cycle protection is included: if the wrapper chain loops, this function
    returns ``False`` rather than looping forever.

    Args:
        env: Environment or wrapper to inspect.
        wrapper_class: Wrapper type to search for.

    Returns:
        ``True`` if an instance of ``wrapper_class`` appears in the wrapper chain;
        ``False`` otherwise.
    """
    current = env
    visited = set()

    while True:
        if id(current) in visited:
            return False
        visited.add(id(current))

        if isinstance(current, wrapper_class):
            return True

        if hasattr(current, "venv"):
            current = current.venv
            continue

        if isinstance(current, gym.Wrapper):
            current = current.env
            continue

        return False

def get_wrapped(env: gym.Env, wrapper_class: gym.Wrapper) -> gym.Env:
    r"""
    Return the first wrapper instance of type ``wrapper_class`` found in ``env``'s
    wrapper chain.

    The traversal rules match :func:`is_wrapped`.

    Args:
        env: Environment or wrapper to inspect.
        wrapper_class: Wrapper type to retrieve.

    Returns:
        The first encountered instance of ``wrapper_class`` in the wrapper chain,
        or ``None`` if it is not present (or if a cycle is detected).
    """

    current = env
    visited = set()

    while True:
        if id(current) in visited:
            return None
        visited.add(id(current))

        if isinstance(current, wrapper_class):
            return current

        if hasattr(current, "venv"):
            current = current.venv
            continue

        if isinstance(current, gym.Wrapper):
            current = current.env
            continue

        return None

class ConstraintPersistentWrapper(gym.Wrapper):
    """
    Base wrapper that *persists* access to constraint-related attributes.

    Many Gymnasium wrappers shadow attributes by changing ``self.env``. This
    wrapper provides stable properties for:

    * :attr:`_constraint` (if present on the underlying env)
    * :attr:`cost_fn` (if exposed by the constraint)
    * :attr:`label_fn` (if present on the underlying env)

    Subclasses can rely on these properties even when stacked with additional
    wrappers.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    @property
    def _constraint(self):
        """
        The underlying constraint object, if present.

        Returns:
            The object stored in ``self.env._constraint`` if it exists, otherwise
            ``None``.
        """
        return getattr(self.env, "_constraint", None)

    @property
    def cost_fn(self):
        """
        Cost function exposed by the underlying constraint, if available.

        If ``self._constraint`` exists and it exposes ``cost_fn``, this returns
        that callable-like object (often a :class:`masa.common.ltl.DFACostFn`).
        Otherwise returns ``None``.

        Returns:
            A cost function-like object or ``None``.
        """
        if self._constraint is not None:
            return getattr(self.env._constraint, "cost_fn", None)
        else:
            return None

    @property
    def label_fn(self):
        """
        Labelling function exposed by the underlying environment, if available.

        Returns:
            The object stored in ``self.env.label_fn`` if it exists, otherwise
            ``None``.
        """
        return getattr(self.env, "label_fn", None)

class ConstraintPersistentObsWrapper(ConstraintPersistentWrapper):
    """
    Base class for wrappers that transform observations while preserving constraint access.

    Subclasses must implement :meth:`_get_obs` which maps raw observations to the
    wrapped observation representation.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def _get_obs(obs: Any) -> Any:
        """
        Transform a raw observation into the wrapped observation.

        Subclasses must implement this.

        Args:
            obs: Raw observation from the underlying environment.

        Returns:
            Transformed observation.

        Raises:
            NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset the environment and transform the returned observation.

        Args:
            seed: Random seed forwarded to the underlying environment.
            options: Reset options forwarded to the underlying environment.

        Returns:
            A tuple ``(obs, info)`` where ``obs`` is transformed via :meth:`_get_obs`.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return self._get_obs(obs), info

    def step(self, action):
        """
        Step the environment and transform the returned observation.

        Args:
            action: Action forwarded to the underlying environment.

        Returns:
            A 5-tuple ``(obs, reward, terminated, truncated, info)`` where ``obs``
            is transformed via :meth:`_get_obs`.
        """
        obs, rew, term, trunc, info = self.env.step(action)
        return self._get_obs(obs), rew, term, trunc, info
        
class TimeLimit(ConstraintPersistentWrapper):
    """
    Episode time-limit wrapper compatible with constraint persistence.

    This is a minimal time-limit wrapper similar in spirit to Gymnasium's
    time-limit handling. It sets the ``truncated`` flag to ``True`` once the
    number of elapsed steps reaches :attr:`_max_episode_steps`.

    Args:
        env: Base environment to wrap.
        max_episode_steps: Maximum number of steps per episode.

    Attributes:
        _max_episode_steps: Configured time limit in steps.
        _elapsed_steps: Counter of steps elapsed in the current episode.
    """

    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        """
        Step the environment and apply time-limit truncation.

        Args:
            action: Action forwarded to the underlying environment.

        Returns:
            A 5-tuple ``(observation, reward, terminated, truncated, info)``. If
            the time limit is reached, ``truncated`` is forced to ``True``.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment and the elapsed step counter.

        Args:
            **kwargs: Forwarded to the underlying environment's ``reset``.

        Returns:
            The underlying environment's ``reset`` return value.
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ConstraintMonitor(ConstraintPersistentWrapper):
    """
    Monitor that injects constraint metadata and metrics into ``info``.

    This wrapper requires the wrapped environment to be a
    :class:`masa.common.constraints.base.BaseConstraintEnv`, so it can query:

    * :attr:`masa.common.constraints.base.BaseConstraintEnv.constraint_type`
    * :meth:`masa.common.constraints.base.BaseConstraintEnv.constraint_step_metrics`
    * :meth:`masa.common.constraints.base.BaseConstraintEnv.constraint_episode_metrics`

    On each step, the wrapper writes:

    * ``info["constraint"]["type"]``: the constraint type string
    * ``info["constraint"]["step"]``: step-level metrics (cheap, safe)
    * ``info["constraint"]["episode"]``: episode-level metrics (when available)

    Args:
        env: Constraint environment to wrap.

    Raises:
        TypeError: If ``env`` is not a :class:`~masa.common.constraints.base.BaseConstraintEnv`.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env, BaseConstraintEnv):  # type: ignore[arg-type]
            raise TypeError(
                "ConstraintMonitor requires env to implement BaseConstraintEnv "
                "(wrap your env with CumulativeCostEnv/StepWiseProbabilisticEnv/...)."
            )
        self._constraint_env: BaseConstraintEnv = env  # type: ignore[assignment]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset and populate initial constraint metadata in ``info``.

        Args:
            seed: Random seed forwarded to the underlying environment.
            options: Reset options forwarded to the underlying environment.

        Returns:
            A tuple ``(obs, info)``. The returned ``info`` includes
            ``info["constraint"]["type"]`` and ``info["constraint"]["step"]``.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info or {})
        info.setdefault("constraint", {})["type"] = self._constraint_env.constraint_type
        info["constraint"]["step"] = self._step_metrics()
        return obs, info

    def step(self, action):
        """
        Step and populate constraint metrics in ``info``.

        Args:
            action: Action forwarded to the underlying environment.

        Returns:
            A 5-tuple ``(observation, reward, terminated, truncated, info)``.
            The returned ``info`` includes ``constraint`` fields described in the
            class docstring.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        info.setdefault("constraint", {})["type"] = self._constraint_env.constraint_type
        info["constraint"]["step"] = self._step_metrics()
        info["constraint"]["episode"] = self._episode_metrics()
        if terminated or truncated:
            info["constraint"]["episode"] = self._episode_metrics()
        return observation, reward, terminated, truncated, info

    def _step_metrics(self) -> Dict[str, float]:
        """
        Read step-level constraint metrics.

        Returns:
            A dictionary of step-level metrics. If the underlying constraint raises
            an exception, returns an empty dictionary.
        """
        try:
            return dict(self._constraint_env.constraint_step_metrics())
        except Exception:
            return {}

    def _episode_metrics(self) -> Dict[str, float]:
        """
        Read episode-level constraint metrics.

        Returns:
            A dictionary of episode-level metrics. If the underlying constraint raises
            an exception, returns an empty dictionary.
        """
        try:
            return dict(self._constraint_env.constraint_episode_metrics())
        except Exception:
            return {}

class RewardMonitor(ConstraintPersistentWrapper):
    """
    Monitor that injects reward/length metrics into ``info``.

    This wrapper tracks:

    * per-step immediate reward in ``info["metrics"]["step"]["reward"]``
    * episode return/length at episode end in ``info["metrics"]["episode"]``

    Args:
        env: Base environment to wrap.

    Attributes:
        total_reward: Accumulated episode reward since last reset.
        total_steps: Number of steps taken since last reset.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset reward counters and forward ``reset`` to the underlying env.

        Args:
            seed: Random seed forwarded to the underlying environment.
            options: Reset options forwarded to the underlying environment.

        Returns:
            A tuple ``(obs, info)`` from the underlying environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.total_reward = 0.0
        self.total_steps = 0
        return obs, info

    def step(self, action):
        """
        Step the environment and update reward metrics.

        Args:
            action: Action forwarded to the underlying environment.

        Returns:
            A 5-tuple ``(observation, reward, terminated, truncated, info)``.
            On episode end, ``info["metrics"]["episode"]`` is populated with
            episode return and length.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self.total_steps += 1
        info = dict(info or {})
        info.setdefault("metrics", {})
        info["metrics"]["step"] = {"reward": reward}
        if terminated or truncated:
            info["metrics"]["episode"] = self._episode_metrics()
        return observation, reward, terminated, truncated, info
        
    def _episode_metrics(self):
        """
        Compute episode-level reward metrics.

        Returns:
            A dictionary with keys:

            * ``"ep_reward"``: total episode reward.
            * ``"ep_length"``: episode length in steps.
        """
        return {"ep_reward": self.total_reward, "ep_length": self.total_steps}

class RewardShapingWrapper(ConstraintPersistentWrapper):
    r"""
    Potential-based reward shaping wrapper for DFA-based safety constraints.

    If the wrapped environment's constraint exposes a :class:`~masa.common.ltl.DFACostFn`,
    this wrapper constructs a shaped cost function :class:`~masa.common.ltl.ShapedCostFn`
    and updates the step ``cost`` entry inside ``info["constraint"]["step"]`` using:

    .. math::

       c'_t \;=\; c_t \;+\; \gamma \Phi(q_{t+1}) \;-\; \Phi(q_t).

    The potential :math:`\Phi` depends on ``impl``:

    * ``"none"``: :math:`\Phi(q)=0` (no shaping)
    * ``"vi"``: approximate value iteration over DFA graph to derive potentials
    * ``"cycle"``: graph-distance based shaping using a reverse-reachability BFS

    Notes:
        This wrapper assumes the wrapped environment is already producing
        ``info["automaton_state"]`` and a constraint monitor-like structure
        ``info["constraint"]["step"]["cost"]``. If these keys are absent, the
        wrapper will fall back to default values (state ``0`` and cost ``0.0``).

    Args:
        env: Base environment to wrap.
        gamma: Discount used in the shaping term :math:`\gamma \Phi(q_{t+1})`.
        impl: Shaping implementation. One of ``{"none", "vi", "cycle"}``.

    Attributes:
        shaped_cost_fn: The cost function exposed by :attr:`cost_fn` after shaping.
        potential_fn: Callable :math:`\Phi(q)` mapping DFA states to potentials.
        _last_potential: Potential at the previous step's DFA state.
        _gamma: Shaping discount factor.
        _impl: Shaping implementation identifier.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99, impl: str = "none"):
        super().__init__(env)
        self._last_potential = 0.0
        self._gamma = gamma
        self._impl = impl

        self._setup_potential_fn()
        self._setup_cost_fn()

    def _setup_cost_fn(self):
        """
        Create :attr:`shaped_cost_fn` if DFA-based constraints are available.

        If the underlying constraint exposes a :class:`~masa.common.ltl.DFACostFn`,
        constructs a :class:`~masa.common.ltl.ShapedCostFn`. Otherwise, uses a
        trivial zero-cost function.

        Returns:
            ``None``. This method sets :attr:`shaped_cost_fn` as a side effect.
        """
        if hasattr(self._constraint, "cost_fn") and isinstance(self._constraint.cost_fn, DFACostFn):
            self.shaped_cost_fn = ShapedCostFn(self._constraint.cost_fn.dfa, self.potential_fn, gamma=self._gamma)
        else:
            self.shaped_cost_fn = lambda q: 0.0

    def _setup_potential_fn(self):
        """
        Construct the potential function :attr:`potential_fn` for shaping.

        For ``impl="none"``, :attr:`potential_fn` is identically zero.

        For ``impl="vi"``, a small fixed number of value-iteration steps are run
        over DFA states, treating accepting states as having a constant terminal
        value and propagating backward through reachable transitions.

        For ``impl="cycle"``, a reverse-graph BFS is used to find a "furthest"
        state from the accepting set and then compute distances to that target,
        yielding a shaping potential based on distance.

        Returns:
            ``None``. This method sets :attr:`potential_fn` and may store
            intermediate tables such as :attr:`V` or :attr:`dist_to_furthest`.

        Raises:
            AssertionError: If ``impl != "none"`` but no DFA-based cost function
            is available on the underlying constraint.
        """

        if self._impl != "none":
            assert hasattr(self._constraint, "cost_fn"), \
            ("RewardShapingWrapper requires env to implement a BaseConstraintEnv that exposes a cost_fn")
            assert isinstance(getattr(self._constraint, "cost_fn", None), DFACostFn), \
            ("RewardShapingWrapper requires env to implement a LTLSafetyEnv with cost_fn class: DFACostFn")

            dfa: DFA = self.env._constraint.cost_fn.dfa
        else:
            self.potential_fn = lambda q: 0.0

        if self._impl == "vi":

            VI_STEPS = 100
            GAMMA = 0.9
            self.V = {q: 0.0 for q in dfa.states}
            assert GAMMA <= self._gamma

            print("Reward shaping DFA ...")
            for i in tqdm(range(VI_STEPS)):
                diff = 0.0
                for u in dfa.states:
                    V_u = self.V[u]
                    self.V[u] = 1.0/(1.0 - GAMMA) if u in dfa.accepting else \
                        np.max([GAMMA * self.V[v] for v in dfa.edges[u].keys()])
                    diff = max(diff, np.abs(V_u - self.V[u]))
                if diff < 1e-6:
                    break

            self.potential_fn = lambda q: self.V[q]

        if self._impl == "cycle":
            
            print("Reward shaping DFA ...")
            edges_rev = {v: set() for v in dfa.states}
            for u in dfa.states:
                if u in dfa.edges:
                    reachable_states = set(dfa.edges[u].keys())
                    for v in reachable_states:
                        edges_rev[v].add(u)

            dist_to_accepting = {u: np.inf for u in dfa.states}
            queue = deque()

            for a in dfa.accepting:
                dist_to_accepting[a] = 0.0
                queue.append(a)

            max_dist = -1
            furthest_state = None

            while queue:
                current = queue.popleft()
                current_dist = dist_to_accepting[current]

                if current_dist > max_dist:
                    max_dist = current_dist
                    furthest_state = current

                for w in edges_rev.get(current, []):
                    if dist_to_accepting[w] == np.inf:
                        dist_to_accepting[w] = current_dist + 1
                        queue.append(w)

            if furthest_state is None:
                furthest_state = dfa.initial

            self.dist_to_furthest = {u: np.inf for u in dfa.states}
            max_finite_dist = 0.0

            if furthest_state is not None:
                u_target = furthest_state
                self.dist_to_furthest[u_target] = 0.0
                queue = deque([u_target])

                while queue:
                    current = queue.popleft()
                    current_dist = self.dist_to_furthest[current]
        
                    if current_dist > max_finite_dist:
                        max_finite_dist = current_dist
                    
                    for w in edges_rev.get(current, []):
                        if self.dist_to_furthest[w] == np.inf:
                            self.dist_to_furthest[w] = current_dist + 1
                            queue.append(w)

            replacement_value = max_finite_dist + 1.0

            for u in dfa.states:
                if self.dist_to_furthest[u] == np.inf:
                    self.dist_to_furthest[u] = replacement_value

            self.potential_fn = lambda q: self.dist_to_furthest[q]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset the environment and initialize shaping state.

        Args:
            seed: Random seed forwarded to the underlying environment.
            options: Reset options forwarded to the underlying environment.

        Returns:
            A tuple ``(obs, info)`` from the underlying environment.

        Notes:
            This wrapper reads ``info["automaton_state"]`` to initialize the
            previous potential :attr:`_last_potential`. If the key is missing,
            it assumes DFA state ``0``.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        automaton_state = info.get("automaton_state", 0)
        self._last_potential = self.potential_fn(automaton_state)
        return obs, info

    def step(self, action: Any):
        """
        Step the environment and apply potential-based shaping to the step cost.

        Args:
            action: Action forwarded to the underlying environment.

        Returns:
            A 5-tuple ``(observation, reward, terminated, truncated, info)``.

        Side effects:
            Updates ``info["constraint"]["step"]["cost"]`` in-place with the shaped
            cost and updates :attr:`_last_potential`.

        Notes:
            If the underlying ``info`` does not contain constraint metrics,
            this method assumes an unshaped step cost of ``0.0`` and will
            still attempt to write back into ``info["constraint"]["step"]``.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        cost = info["constraint"]["step"].get("cost", 0.0)
        automaton_state = info.get("automaton_state", 0)
        potential = self.potential_fn(automaton_state)
        info["constraint"]["step"]["cost"] = cost + self._gamma * potential - self._last_potential
        self._last_potential = potential
        return observation, reward, terminated, truncated, info

    @property
    def cost_fn(self):
        """
        Expose the shaped cost function.

        Returns:
            The shaped cost function constructed in :meth:`_setup_cost_fn`.
        """
        return self.shaped_cost_fn

class NormWrapper(ConstraintPersistentWrapper):
    """
    Normalize observations and/or rewards for a *single* (non-vectorized) environment.

    This wrapper maintains running mean/variance estimates and applies:

    * Observation normalization (elementwise): :math:`(x - \\mu) / \\sqrt{\\sigma^2 + \\varepsilon}`
    * Reward normalization using a running variance estimate over discounted returns.

    This wrapper is intended for non-vectorized environments. For vectorized
    environments, use :class:`VecNormWrapper`.

    Args:
        env: Base (non-vectorized) environment.
        norm_obs: Whether to normalize observations.
        norm_rew: Whether to normalize rewards.
        training: If ``True``, update running statistics; otherwise, statistics are frozen.
        clip_obs: Clip normalized observations to ``[-clip_obs, clip_obs]``.
        clip_rew: Clip normalized rewards to ``[-clip_rew, clip_rew]``.
        gamma: Discount factor for the running return used in reward normalization.
        eps: Small constant :math:`\\varepsilon` for numerical stability.

    Attributes:
        norm_obs: See Args.
        norm_rew: See Args.
        training: See Args.
        clip_obs: See Args.
        clip_rew: See Args.
        gamma: See Args.
        eps: See Args.
        obs_rms: :class:`masa.common.running_mean_std.RunningMeanStd` for observations.
        rew_rms: :class:`masa.common.running_mean_std.RunningMeanStd` for returns.
        returns: Discounted return accumulator used for reward normalization.
    """

    def __init__(
        self, 
        env: gym.Env, 
        norm_obs: bool = True,
        norm_rew: bool = True,
        training: bool = True,
        clip_obs: float = 10.0,
        clip_rew: float = 10.0,
        gamma: float = 0.99,
        eps: float = 1e-8
    ):
        assert not isinstance(
            env, VecEnvWrapperBase
        ), "NormWrapper does not expect a vectorized environment (DummyVecWrapper / VecWrapper). Please use VecNormWrapper instead"

        assert norm_obs and isinstance(
            env.observation_space, spaces.Box
        ), "NormWrapper only supports Box observation spaces when norm_obs=True."

        super().__init__(env)

        self.norm_obs = norm_obs
        self.norm_rew = norm_rew
        self.training = training
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew
        self.gamma = gamma
        self.eps = eps

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.rew_rms = RunningMeanStd(shape=())

        self.returns = np.zeros(1, dtype=np.float32)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize (and clip) a single observation.

        Args:
            obs: Raw observation.

        Returns:
            Normalized observation.
        """
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps),
            -self.clip_obs,
            self.clip_obs
        )

    def _normalize_rew(self, rew: float) -> float:
        """
        Normalize (and clip) a single reward.

        Args:
            rew: Raw reward.

        Returns:
            Normalized reward.
        """
        return np.clip(
            rew / np.sqrt(self.rew_rms.var + self.eps),
            -self.clip_rew,
            self.clip_rew,
        )

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset the environment and (optionally) update normalization statistics.

        Args:
            seed: Random seed forwarded to the underlying environment.
            options: Reset options forwarded to the underlying environment.

        Returns:
            A tuple ``(obs, info)`` where ``obs`` may be normalized.
        """
        obs, info = self.env.reset(seed=seed, options=options)

        if self.norm_obs and self.training:
            self.obs_rms.update(obs)

        if self.norm_rew and self.training:
            self.returns[:] = 0.0

        if self.norm_obs:
            obs = self._normalize_obs(obs)

        return obs, info

    def step(self, action):
        """
        Step the environment and apply observation/reward normalization.

        Args:
            action: Action forwarded to the underlying environment.

        Returns:
            A 5-tuple ``(obs, rew, terminated, truncated, info)`` where ``obs`` and/or
            ``rew`` may be normalized.
        """
        obs, rew, term, trunc, info = self.env.step(action)

        if self.norm_obs and self.training:
            self.obs_rms.update(obs)

        if self.norm_rew:
            self.returns = self.returns * self.gamma + rew
            if self.training:
                self.rew_rms.update(self.returns)

            rew = self._normalize_rew(rew)

        if self.norm_obs:
            obs = self._normalize_obs(obs)

        return obs, rew, term, trunc, info

class OneHotObsWrapper(ConstraintPersistentObsWrapper):
    """
    One-hot encode :class:`gymnasium.spaces.Discrete` observations.

    Supported input observation spaces:

    * :class:`gymnasium.spaces.Discrete`: returns a 1D one-hot vector of length ``n``.
    * :class:`gymnasium.spaces.Dict`: one-hot encodes any Discrete subspaces and
      passes through non-Discrete subspaces.
    * Otherwise: passes observations through unchanged.

    The wrapper updates :attr:`gymnasium.Env.observation_space` accordingly.

    Args:
        env: Base environment to wrap.

    Attributes:
        _orig_obs_space: The original observation space of the wrapped env.
        _mode: One of ``{"discrete", "dict", "pass"}`` describing the encoding mode.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._orig_obs_space = self.env.observation_space

        if isinstance(self._orig_obs_space, spaces.Discrete):
            self._mode = "discrete"
            n = self._orig_obs_space.n
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n,),
                dtype=np.float32,
            )
            

        elif isinstance(self._orig_obs_space, spaces.Dict):
            self._mode = "dict"

            new_spaces: Dict[str, spaces.Space] = {}
            for key, subspace in self._orig_obs_space.spaces.items():
                if isinstance(subspace, spaces.Discrete):
                    n = subspace.n
                    new_spaces[key] = spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(n,),
                        dtype=np.float32,
                    )
                else:
                    # Preserve non-Discrete subspace as-is
                    new_spaces[key] = subspace

            self.observation_space = spaces.Dict(new_spaces)

        else:
            self._mode = "pass"
            self.observation_space = self._orig_obs_space

    @staticmethod
    def _one_hot_scalar(idx: int, n: int) -> np.ndarray:
        """
        One-hot encode an integer index.

        Args:
            idx: Index in ``{0, 1, ..., n-1}``.
            n: Vector length.

        Returns:
            A float32 vector ``v`` with ``v[idx] = 1`` and zeros elsewhere.
        """
        one_hot = np.zeros(n, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def _get_obs(self, obs: Union[int, Dict[str, Any], np.ndarray]) -> np.ndarray:
        """
        Transform an observation according to the wrapper's configured mode.

        Args:
            obs: Raw observation.

        Returns:
            One-hot encoded observation (or dict containing one-hot fields) when applicable,
            otherwise the original observation.
        """
        if self._mode == "discrete":
            # Original obs_space is Discrete; obs is an int-like
            idx = int(obs)
            n = self._orig_obs_space.n
            return self._one_hot_scalar(idx, n)

        elif self._mode == "dict":
            assert isinstance(obs, dict), (
                f"Expected dict observation for Dict space, got {type(obs)}"
            )

            new_obs: Dict[str, Any] = {}
            for key, subspace in self._orig_obs_space.spaces.items():
                value = obs[key]

                if isinstance(subspace, spaces.Discrete):
                    idx = int(value)
                    new_obs[key] = self._one_hot_scalar(idx, subspace.n)
                else:
                    # Leave non-Discrete parts unchanged
                    new_obs[key] = value

            return new_obs
        else: 
            # pass
            return obs

class FlattenDictObsWrapper(ConstraintPersistentObsWrapper):
    """
    Flatten a :class:`gymnasium.spaces.Dict` observation into a 1D :class:`~gymnasium.spaces.Box`.

    The wrapper creates a deterministic key ordering (alphabetical) and concatenates
    each sub-observation in that order.

    Supported Dict subspaces:

    * :class:`gymnasium.spaces.Box`: flattened via ``reshape(-1)``.
    * :class:`gymnasium.spaces.Discrete`: represented as a length-``n`` one-hot
      segment for the purposes of bounds (note: the current implementation of
      :meth:`_get_obs` expects Box values; see Notes).

    Args:
        env: Base environment with Dict observation space.

    Attributes:
        _orig_obs_space: Original Dict observation space.
        _key_slices: Mapping from key to slice in the flattened vector.

    Raises:
        TypeError: If the underlying observation space is not a Dict, or contains
            unsupported subspaces.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._orig_obs_space = self.env.observation_space

        if not isinstance(self._orig_obs_space, spaces.Dict):
            raise TypeError(
                f"FlattenDictObsWrapper requires Dict observation space, got {type(self._orig_obs_space)}"
            )

        # To be able to reconstruct if needed, keep slices for each key
        self._key_slices: dict[str, slice] = {}

        low_parts = []
        high_parts = []
        offset = 0

        # Sort keys alphabetically for deterministic ordering
        for key in sorted(self._orig_obs_space.spaces.keys()):
            subspace = self._orig_obs_space.spaces[key]

            if isinstance(subspace, spaces.Box):
                # Flatten Box
                low = np.asarray(subspace.low, dtype=np.float32).reshape(-1)
                high = np.asarray(subspace.high, dtype=np.float32).reshape(-1)
                length = low.shape[0]

                low_parts.append(low)
                high_parts.append(high)

            elif isinstance(subspace, spaces.Discrete):
                # One-hot will be in [0, 1]
                length = subspace.n
                low_parts.append(np.zeros(length, dtype=np.float32))
                high_parts.append(np.ones(length, dtype=np.float32))

            else:
                raise TypeError(
                    f"Unsupported subspace type for key '{key}': {type(subspace)}"
                )

            self._key_slices[key] = slice(offset, offset + length)
            offset += length

        low = np.concatenate(low_parts).astype(np.float32)
        high = np.concatenate(high_parts).astype(np.float32)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def _get_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Flatten a Dict observation into a 1D vector.

        Args:
            obs: Dict observation keyed the same way as the original Dict space.

        Returns:
            A 1D float32 array created by concatenating flattened sub-observations.

        Raises:
            TypeError: If any subspace is not a Box.

        Notes:
            Although the constructor supports Discrete subspaces when building bounds,
            this implementation currently enforces Box-only subspaces at runtime.
        """
        assert isinstance(obs, dict), (
                f"Expected dict observation for Dict space, got {type(obs)}"
            )

        parts = []
        for key in sorted(self._orig_obs_space.spaces.keys()):
            subspace = self._orig_obs_space.spaces[key]
            value = obs[key]

            if not isinstance(subspace, spaces.Box):
                raise TypeError(
                    f"FlattenDictObsWrapper only supports Box subspaces, "
                    f"got {type(subspace)} for key '{key}'"
                )

            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            parts.append(arr)

        return np.concatenate(parts).astype(np.float32)

class VecEnvWrapperBase(ConstraintPersistentWrapper):
    """
    Base class for simple Python-list vector environment wrappers.

    Vector environments in this file expose:

    * :attr:`n_envs`: number of parallel environments
    * :meth:`reset`: returns ``(obs_list, info_list)``
    * :meth:`step`: returns ``(obs_list, rew_list, term_list, trunc_list, info_list)``
    * :meth:`reset_done`: reset only environments indicated by a ``dones`` mask

    Args:
        env: For :class:`DummyVecWrapper`, this is the single underlying env.
            For :class:`VecWrapper`, this is set to ``envs[0]`` to preserve a
            Gymnasium-like API surface.

    Attributes:
        n_envs: Number of environments.
    """

    n_envs: int

    def __init__(self, env: gym.Env):
        # For DummyVecWrapper: env is the single env
        # For VecWrapper: env is envs[0]
        # For VecNormWrapper: env is a VecEnvWrapperBase
        super().__init__(env)

    def reset_done(
        self, 
        dones: Union[List[bool], np.ndarray],
        *, 
        seed: int | None = None, 
        options: Dict[str, Any] | None = None
    ):
        """
        Reset only the environments indicated by ``dones``.

        Args:
            dones: Boolean mask/list of length :attr:`n_envs`. Entries set to
                ``True`` are reset.
            seed: Optional base seed. Implementations may offset by environment index.
            options: Reset options forwarded to underlying environments.

        Returns:
            A tuple ``(reset_obs, reset_infos)`` where:

            * ``reset_obs`` is a list of length :attr:`n_envs` containing reset
              observations at indices that were reset, and ``None`` elsewhere.
            * ``reset_infos`` is a list of length :attr:`n_envs` containing reset
              info dicts at indices that were reset, and empty dicts elsewhere.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError
        
class DummyVecWrapper(VecEnvWrapperBase):
    """
    Wrap a single environment with a vector-environment API (``n_envs=1``).

    This wrapper is useful for code paths that expect list-based vector outputs,
    while still running a single environment instance.

    Args:
        env: Base environment.

    Attributes:
        n_envs: Always ``1``.
        envs: List containing the single wrapped environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_envs = 1
        self.envs: List[gym.Env] = [env]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset and return vectorized lists of length 1.

        Args:
            seed: Random seed forwarded to the underlying environment.
            options: Reset options forwarded to the underlying environment.

        Returns:
            ``([obs], [info])``.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return [obs], [info]

    def reset_done(
        self, 
        dones: Union[List[bool], np.ndarray],
        *, 
        seed: int | None = None, 
        options: Dict[str, Any] | None = None
    ):
        """
        Conditionally reset the single environment.

        Args:
            dones: A length-1 mask. If ``dones[0]`` is ``True``, reset.
            seed: Random seed forwarded to the underlying environment.
            options: Reset options forwarded to the underlying environment.

        Returns:
            A pair ``(reset_obs, reset_infos)`` as described by
            :meth:`VecEnvWrapperBase.reset_done`.
        """
        dones = list(dones)
        assert len(dones) == 1
        if dones[0]:
            return self.reset(seed=seed, options=options)
        else:
            [None], [{}]

    def step(self, action):
        """
        Step and return vectorized lists of length 1.

        Args:
            action: Action for the single environment.

        Returns:
            ``([obs], [rew], [terminated], [truncated], [info])``.
        """
        obs, rew, term, trunc, info = self.env.step(action)
        return [obs], [rew], [term], [trunc], [info]

class VecWrapper(VecEnvWrapperBase):
    """
    Wrap a list of environments with a simple vector-environment API.

    Each underlying environment is reset/stepped sequentially in Python, and
    results are returned as Python lists.

    Args:
        envs: Non-empty list of environments.

    Attributes:
        envs: The list of wrapped environments.
        n_envs: Number of wrapped environments.
    """

    def __init__(self, envs: List[gym.Env]):
        assert len(envs) > 0, "VecWrapper requires at least one environment"
        super().__init__(envs[0]) # maintain API compatibility
        self.envs: List[gym.Env] = envs
        self.n_envs = len(envs)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset all environments and return lists.

        Args:
            seed: Optional base seed. If provided, environment ``i`` receives ``seed + i``.
            options: Reset options forwarded to each environment.

        Returns:
            A pair ``(obs_list, info_list)`` of length :attr:`n_envs`.
        """
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + i
            obs, info = env.reset(seed=s, options=options)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def reset_done(
        self, 
        dones: Union[List[bool], np.ndarray],
        *, 
        seed: int | None = None, 
        options: Dict[str, Any] | None = None
    ):
        """
        Reset only environments whose done flag is True.

        Args:
            dones: Boolean mask/list of length :attr:`n_envs`.
            seed: Optional base seed. If provided, environment ``i`` receives ``seed + i``.
            options: Reset options forwarded to environments being reset.

        Returns:
            A tuple ``(reset_obs, reset_infos)`` where non-reset indices contain
            ``None`` and ``{}`` respectively.
        """
        dones = list(dones)
        assert len(dones) == self.n_envs

        reset_obs = [None] * self.n_envs
        reset_infos = [{} for _ in range(self.n_envs)]

        for i, done in enumerate(dones):
            if done:
                s = None if seed is None else seed + i
                obs, info = self.envs[i].reset(seed=s, options=options)
                reset_obs[i] = obs
                reset_infos[i] = info

        return reset_obs, reset_infos

    def step(self, actions):
        """
        Step all environments.

        Args:
            actions: Iterable of actions of length :attr:`n_envs`.

        Returns:
            A 5-tuple of lists ``(obs_list, rew_list, term_list, trunc_list, info_list)``.

        Notes:
            The loop expects one action per environment. If the provided
            ``actions`` length mismatches :attr:`n_envs`, Python will raise.
        """
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, rew, term, trunc, info = env.step(action)
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)

        return obs_list, rew_list, term_list, trunc_list, info_list


class VecNormWrapper(VecEnvWrapperBase):
    """
    Normalize observations and/or rewards for a vectorized environment.

    This wrapper expects an environment implementing :class:`VecEnvWrapperBase`
    (e.g., :class:`DummyVecWrapper` or :class:`VecWrapper`) and applies the same
    normalization logic as :class:`NormWrapper`, but over batches.

    Observation normalization uses running statistics of the stacked observation
    array (shape ``(n_envs, *obs_shape)``). Reward normalization uses running
    statistics of discounted returns per environment.

    Args:
        env: A vectorized environment implementing :class:`VecEnvWrapperBase`.
        norm_obs: Whether to normalize observations.
        norm_rew: Whether to normalize rewards.
        training: If ``True``, update running statistics; otherwise, statistics are frozen.
        clip_obs: Clip normalized observations to ``[-clip_obs, clip_obs]``.
        clip_rew: Clip normalized rewards to ``[-clip_rew, clip_rew]``.
        gamma: Discount factor for the running return used in reward normalization.
        eps: Small constant :math:`\\varepsilon` for numerical stability.

    Attributes:
        n_envs: Copied from the wrapped vector environment.
        obs_rms: :class:`masa.common.running_mean_std.RunningMeanStd` for observations.
        rew_rms: :class:`masa.common.running_mean_std.RunningMeanStd` for returns.
        returns: Vector of length :attr:`n_envs` storing per-env discounted returns.
    """

    def __init__(
        self, 
        env: Union[gym.Env, List[gym.Env]], 
        norm_obs: bool = True,
        norm_rew: bool = True,
        training: bool = True,
        clip_obs: float = 10.0,
        clip_rew: float = 10.0,
        gamma: float = 0.99,
        eps: float = 1e-8
    ):
        assert isinstance(
            env, VecEnvWrapperBase
        ), "VecNormWrapper expects a vectorized environment (DummyVecWrapper / VecWrapper)."

        assert norm_obs and isinstance(
            env.observation_space, spaces.Box
        ), "VecNormWrapper only supports Box observation spaces when norm_obs=True."

        super().__init__(env)

        self.n_envs = env.n_envs
        self.norm_obs = norm_obs
        self.norm_rew = norm_rew
        self.training = training
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew
        self.gamma = gamma
        self.eps = eps

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.rew_rms = RunningMeanStd(shape=())

        self.returns = np.zeros(self.n_envs, dtype=np.float32)

    def _normalize_obs(self, obs_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize and clip a list of observations.

        Args:
            obs_list: List of raw observations of length :attr:`n_envs`.

        Returns:
            List of normalized observations.
        """
        obs_arr = np.asarray(obs_list, dtype=np.float32)
        norm = (obs_arr - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)
        norm = np.clip(norm, -self.clip_obs, self.clip_obs)
        return norm.tolist()

    def _normalize_rew(self, rew_list: List[float]) -> List[float]:
        """
        Normalize and clip a list of rewards.

        Args:
            rew_list: List of raw rewards of length :attr:`n_envs`.

        Returns:
            List of normalized rewards.
        """
        rew_arr = np.asarray(rew_list, dtype=np.float32)
        norm = rew_arr / np.sqrt(self.rew_rms.var + self.eps)
        norm = np.clip(norm, -self.clip_rew, self.clip_rew)
        return norm.tolist()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset all environments and normalize observations.

        Args:
            seed: Optional base seed forwarded to the underlying vector env.
            options: Reset options forwarded to the underlying vector env.

        Returns:
            A pair ``(obs_list, info_list)``. Observations may be normalized.
        """
        obs_list, info_list = self.env.reset(seed=seed, options=options)

        if self.norm_obs and self.training:
            self.obs_rms.update(np.asarray(obs_list, dtype=np.float32))

        self.returns[:] = 0.0

        if self.norm_obs:
            obs_list = self._normalize_obs(obs_list)

        return obs_list, info_list

    def reset_done(
        self, 
        dones: Union[List[bool], np.ndarray],
        *, 
        seed: int | None = None, 
        options: Dict[str, Any] | None = None
    ):
        """
        Reset only environments indicated by ``dones`` and normalize those observations.

        Args:
            dones: Boolean mask/list of length :attr:`n_envs`.
            seed: Optional base seed forwarded to the underlying vector env.
            options: Reset options forwarded to the underlying vector env.

        Returns:
            A tuple ``(reset_obs, reset_infos)`` as described by
            :meth:`VecEnvWrapperBase.reset_done`, with reset observations optionally
            normalized.
        """
        reset_obs, reset_infos = self.env.reset_done(
            dones, seed=seed, options=options
        )

        obs_arr = np.asarray(
            [o for o in reset_obs if o is not None],
            dtype=np.float32,
        ) if any(o is not None for o in reset_obs) else None

        if self.norm_obs and self.training and obs_arr is not None:
            self.obs_rms.update(obs_arr)

        for i, done in enumerate(dones):
            if done:
                self.returns[i] = 0.0

        if self.norm_obs:
            norm_reset_obs: List[Any] = list(reset_obs)
            # Only normalize indices that were reset
            for i, done in enumerate(dones):
                if done and reset_obs[i] is not None:
                    o = np.asarray(reset_obs[i], dtype=np.float32)
                    norm = (o - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)
                    norm = np.clip(norm, -self.clip_obs, self.clip_obs)
                    norm_reset_obs[i] = norm
            reset_obs = norm_reset_obs

        return reset_obs, reset_infos

    def step(self, actions):
        """
        Step all environments and apply observation/reward normalization.

        Args:
            actions: Iterable of actions of length :attr:`n_envs`.

        Returns:
            A 5-tuple ``(obs_list, rew_list, term_list, trunc_list, infos)``, where
            observations and/or rewards may be normalized.
        """
        obs_list, rew_list, term_list, trunc_list, infos = self.env.step(actions)

        obs_arr = np.asarray(obs_list, dtype=np.float32)
        rew_arr = np.asarray(rew_list, dtype=np.float32)

        if self.norm_obs and self.training:
            self.obs_rms.update(obs_arr)

        if self.norm_rew:
            self.returns = self.returns * self.gamma + rew_arr
            if self.training:
                self.rew_rms.update(self.returns)

            rew_list = self._normalize_rew(rew_list)

        if self.norm_obs:
            obs_list = self._normalize_obs(obs_list)

        return obs_list, rew_list, term_list, trunc_list, infos

        