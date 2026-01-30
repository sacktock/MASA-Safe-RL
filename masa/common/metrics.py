from __future__ import annotations

import time
import warnings
from collections import deque
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Stats:
    r"""Streaming scalar summary statistics for an arbitrary batch of values.

    This class maintains simple running aggregates over a stream of scalar
    observations:

    - Mean: :math:`\mu`
    - Mean of squares: :math:`\mathbb{E}[X^2]`
    - Derived standard deviation: :math:`\sigma = \sqrt{\max(0, \mathbb{E}[X^2] - \mu^2)}`
    - Extremes: min/max
    - Magnitude: :math:`\max |x|`

    The :meth:`update` method accepts any array-like input, flattens it to a 1D
    vector, and updates the aggregates. The :meth:`get` method returns a dict of
    scalars suitable for logging; if :attr:`prefix` is non-empty, keys are
    prefixed with ``"{prefix}_"``.

    Args:
        prefix: Optional key prefix (without trailing underscore). If provided,
            keys returned by :meth:`get` are of the form ``"{prefix}_{name}"``.

    Attributes:
        n: Total number of scalar samples seen so far.
        prefix: Prefix used to namespace emitted keys.
        stats: Internal aggregate state or ``None`` if no data has been observed.
            When populated, it contains keys ``mean``, ``mean_squares``, ``max``,
            ``min``, and ``mag``.
    """

    def __init__(self, prefix: str = ""):
        self.n: int = 0
        self.prefix: str = prefix
        self.stats: Optional[Dict[str, float]] = None

    def update(self, values: Union[np.ndarray, Iterable[float], float]):
        """Update aggregates with a batch of scalar values.

        The input is converted to ``np.float32`` and flattened. If the resulting
        array is empty, the call is a no-op.

        Args:
            values: A scalar or array-like collection of numeric values.
        """
        v = np.asarray(values, dtype=np.float32).ravel()
        m = int(v.size)
        if m == 0:
            return

        if self.stats is None:
            self.n = m
            self.stats = {
                "mean": float(np.mean(v)),
                "mean_squares": float(np.mean(v**2)),
                "max": float(np.max(v)),
                "min": float(np.min(v)),
                "mag": float(np.max(np.abs(v))),
            }
            return

        # Weighted update of first and second moments.
        n_prev = self.n
        self.n += m
        w_new = m / self.n
        w_old = (self.n - m) / self.n

        self.stats = {
            "mean": float(np.mean(v)) * w_new + float(self.stats["mean"]) * w_old,
            "mean_squares": float(np.mean(v**2)) * w_new
            + float(self.stats["mean_squares"]) * w_old,
            "max": float(max(float(np.max(v)), float(self.stats["max"]))),
            "min": float(min(float(np.min(v)), float(self.stats["min"]))),
            "mag": float(max(float(np.max(np.abs(v))), float(self.stats["mag"]))),
        }

    def get(self) -> Dict[str, float]:
        r"""Return the current statistics as a flat dict of scalars.

        Returns:
            A mapping containing:

            - ``mean``: :math:`\mu`
            - ``std``: :math:`\sigma = \sqrt{\max(0, \mathbb{E}[X^2] - \mu^2)}`
            - ``max``: :math:`\max x`
            - ``min``: :math:`\min x`
            - ``mag``: :math:`\max |x|`

            If :attr:`prefix` is non-empty, keys are prefixed with
            ``"{prefix}_"``.

        Raises:
            RuntimeError: If called before any data has been observed.
        """
        if self.stats is None:
            raise RuntimeError("Stats.get() called before any update().")

        mean = float(self.stats["mean"])
        mean_sq = float(self.stats["mean_squares"])
        std = float(np.sqrt(np.maximum(0.0, mean_sq - mean**2)))

        out: Dict[str, float] = {
            "mean": mean,
            "std": std,
            "max": float(self.stats["max"]),
            "min": float(self.stats["min"]),
            "mag": float(self.stats["mag"]),
        }
        if self.prefix:
            out = {f"{self.prefix}_{k}": v for k, v in out.items()}
        return out

    def __add__(self, other: "Stats") -> "Stats":
        """Combine two :class:`Stats` objects into a new aggregate.

        This is useful when you have two independent streams and want the same
        summary you would have obtained if you had processed all samples in one
        stream. The combination is exact for the tracked aggregates
        (first/second moments and extrema).

        Args:
            other: Another :class:`Stats` instance with the same :attr:`prefix`.

        Returns:
            A new :class:`Stats` instance representing the combined aggregate.

        Raises:
            AssertionError: If :attr:`prefix` differs.
            RuntimeError: If either object has not observed any data.
        """
        assert isinstance(other, Stats)
        assert self.prefix == other.prefix, "can't add two Stats objects with different prefixes"
        if self.stats is None or other.stats is None:
            raise RuntimeError("Cannot add Stats objects before both have been updated at least once.")

        new = Stats(prefix=self.prefix)
        new.n = int(self.n + other.n)

        w_self = self.n / new.n
        w_other = other.n / new.n
        new.stats = {
            "mean": float(self.stats["mean"]) * w_self + float(other.stats["mean"]) * w_other,
            "mean_squares": float(self.stats["mean_squares"]) * w_self
            + float(other.stats["mean_squares"]) * w_other,
            "max": float(max(float(self.stats["max"]), float(other.stats["max"]))),
            "min": float(min(float(self.stats["min"]), float(other.stats["min"]))),
            "mag": float(max(float(self.stats["mag"]), float(other.stats["mag"]))),
        }
        return new

class Dist:
    r"""Reservoir-sampled distribution summary.

    This class maintains a fixed-size reservoir sample of a stream of scalar
    values using reservoir sampling. After processing :math:`n` total samples, a
    reservoir of size :math:`k` contains a uniform sample (without replacement)
    from the observed stream (in expectation).

    Args:
        prefix: Optional logical prefix for this distribution (used by loggers).
        reservoir_size: Maximum number of samples retained in the reservoir.
        rng: Seed (or seed-like) passed to :func:`numpy.random.default_rng`.

    Attributes:
        n: Total number of scalar samples seen so far.
        prefix: Prefix used by :class:`StatsLogger` when naming histogram keys.
        reservoir_size: Maximum reservoir capacity.
        res: Current reservoir buffer of shape ``(<=reservoir_size,)``.
        rng: Numpy random generator used for reservoir sampling.
    """

    def __init__(
        self,
        prefix: str = "",
        reservoir_size: int = 2048,
        rng: Optional[Union[int, np.random.Generator, np.random.BitGenerator]] = None,
    ):
        self.n: int = 0
        self.prefix: str = prefix
        self.reservoir_size: int = int(reservoir_size)
        self.res: np.ndarray = np.empty((0,), dtype=np.float32)
        self.rng: np.random.Generator = np.random.default_rng(rng)

    def update(self, values: Union[np.ndarray, Iterable[float], float]):
        """Update the reservoir with a batch of values.

        Args:
            values: A scalar or array-like collection of numeric values. The
                input is flattened into a 1D array.
        """
        v = np.asarray(values, dtype=np.float32).ravel()
        m = int(v.size)
        if m == 0:
            return

        # Fill reservoir initially.
        if self.n < self.reservoir_size:
            take = min(self.reservoir_size - self.n, m)
            if take > 0:
                self.res = np.concatenate([self.res, v[:take]])
                self.n += int(take)
                v = v[take:]

        # Reservoir sampling for the remainder.
        for x in v:
            self.n += 1
            j = int(self.rng.integers(0, self.n))
            if j < self.reservoir_size:
                self.res[j] = x

    def get(self) -> np.ndarray:
        """Return a copy of the reservoir buffer.

        Returns:
            A copy of the internal reservoir array (dtype ``np.float32``).
        """
        return self.res.copy()

    def __add__(self, other: "Dist") -> "Dist":
        """Merge two reservoirs into a new reservoir.

        The merged reservoir is produced by sampling (without replacement) from
        the concatenation of both reservoirs, capped at :attr:`reservoir_size`.
        This is a pragmatic merge for logging/visualisation; it does not exactly
        reproduce a single-pass reservoir over the full underlying streams.

        Args:
            other: Another :class:`Dist` with the same :attr:`prefix` and
                :attr:`reservoir_size`.

        Returns:
            A new :class:`Dist` containing a merged reservoir sample.

        Raises:
            AssertionError: If :attr:`prefix` differs.
        """
        assert isinstance(other, Dist)
        assert self.prefix == other.prefix, "can't add two Dist objects with different prefixes"
        assert (
            self.reservoir_size == other.reservoir_size
        ), "can't add two Dist objects with different reservoir sizes"

        new = Dist(prefix=self.prefix, reservoir_size=self.reservoir_size, rng=0)
        new.n = int(self.n + other.n)

        both = np.concatenate([self.res, other.res])
        if both.size <= new.reservoir_size:
            new.res = both.astype(np.float32, copy=False)
        else:
            idx = np.random.default_rng(0).choice(
                both.size, size=new.reservoir_size, replace=False
            )
            new.res = both[idx].astype(np.float32, copy=False)
        return new

class BaseLogger:
    r"""Base class for logging scalar statistics and distributions.

    A logger ingests objects via :meth:`add` and produces aggregated outputs via
    :meth:`log`. Concrete subclasses define how they ingest and summarise data.

    TensorBoard logging uses a :class:`tf.summary.SummaryWriter`. If
    :attr:`tensorboard` is ``True`` then a writer must be provided.

    Args:
        stdout: If ``True``, print logs to stdout via :func:`print` or
            :func:`tqdm.write`.
        tqdm: If ``True``, use :func:`tqdm.write` for stdout printing.
        tensorboard: If ``True``, emit TensorBoard summaries.
        summary_writer: TensorBoard writer. Required when ``tensorboard=True``.
        stats_window_size: Maximum number of recent scalar values retained per
            metric.
        prefix: Optional string prefix for TensorBoard tag names and stdout
            display. If non-empty, a trailing ``"/"`` is ensured.

    Attributes:
        stdout: Whether stdout logging is enabled.
        tqdm: Whether tqdm-compatible printing is enabled.
        tensorboard: Whether TensorBoard logging is enabled.
        summary_writer: TensorBoard summary writer (may be ``None``).
        stats_window_size: Window size for scalar smoothing.
        prefix: Namespace prefix ending with ``"/"`` (or empty).
        stats: Mapping from metric key to a deque of recent values.
    """

    def __init__(
        self,
        stdout: bool = True,
        tqdm: bool = True,
        tensorboard: bool = False,
        summary_writer: Optional[tf.summary.SummaryWriter] = None,
        stats_window_size: int = 100,
        prefix: str = "",
    ):
        self.stdout: bool = stdout
        self.tqdm: bool = tqdm
        self.tensorboard: bool = tensorboard
        self.summary_writer: Optional[tf.summary.SummaryWriter] = summary_writer

        if self.tensorboard:
            assert self.summary_writer is not None, "tensorboard=True requires a summary_writer"
        if (self.summary_writer is not None) and (not self.tensorboard):
            warnings.warn(
                "tensorboard is set to False but summary writer is provided; this may produce unexpected behaviour",
                stacklevel=2,
            )

        self.stats_window_size: int = int(stats_window_size)
        self.prefix: str = (prefix if not prefix else (prefix if prefix.endswith("/") else prefix + "/"))
        self.stats: Dict[str, Deque[float]] = {}

    def reset(self):
        """Clear all buffered statistics."""
        self.stats = {}

    def add(self, new: Any):
        """Ingest a new object into the logger.

        Concrete subclasses define the supported input types.

        Args:
            new: Object to ingest.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def log(self, step: int):
        """Emit logs for a given global step.

        Args:
            step: Global step index used for TensorBoard summary steps.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

class StatsLogger(BaseLogger):
    r"""Logger for streaming scalar stats (:class:`Stats`) and distributions (:class:`Dist`).

    The :meth:`add` method accepts a mapping whose values are one of:

    - :class:`Stats`: expanded into multiple scalar keys (mean/std/min/max/mag).
    - :class:`Dist`: captured for histogram logging.
    - ``float`` / ``int`` / numpy scalar: treated as a scalar time series.

    Aggregation:
        - Scalars are smoothed by taking the mean of the most recent
          :attr:`~BaseLogger.stats_window_size` values.
        - Distributions are logged as histograms using the stored reservoir.

    Notes:
        This class creates internal dictionaries :attr:`stats_to_log` and
        :attr:`dists_to_log` during :meth:`log`.
    """

    def __init__(
        self,
        stdout: bool = True,
        tqdm: bool = True,
        tensorboard: bool = False,
        summary_writer: Optional[tf.summary.SummaryWriter] = None,
        stats_window_size: int = 100,
        prefix: str = "",
    ):
        super().__init__(
            stdout=stdout,
            tqdm=tqdm,
            tensorboard=tensorboard,
            summary_writer=summary_writer,
            stats_window_size=stats_window_size,
            prefix=prefix,
        )
        self.dists: Dict[str, np.ndarray] = {}
        self.stats_to_log: Dict[str, float] = {}
        self.dists_to_log: Dict[str, np.ndarray] = {}


    def reset(self):
        """Clear all buffered scalar and distribution values."""
        super().reset()
        self.dists = {}
        self.stats_to_log = {}
        self.dists_to_log = {}

    def add(self, new: Mapping[str, Union["Stats", "Dist", float, int, np.floating]]):
        """Add a batch of metrics to the logger.

        Args:
            new: Mapping from metric name to a supported metric object.

        Raises:
            NotImplementedError: If a value type is unsupported.
        """

        for key, val in new.items():
            if isinstance(val, Stats):
                met = val.get()
                for k, v in met.items():
                    if f"{key}_{k}" in self.stats:
                        self.stats[f"{key}_{k}"].append(float(v))
                    else:
                        self.stats[f"{key}_{k}"] = deque([float(v)], maxlen=self.stats_window_size)
            elif isinstance(val, Dist):
                # Store a snapshot of the reservoir for later histogram logging.
                self.dists[key] = val.get()
            elif isinstance(val, (float, int, np.floating)):
                if key in self.stats:
                    self.stats[key].append(float(val))
                else:
                    self.stats[key] = deque([float(val)], maxlen=self.stats_window_size)
            else:
                raise NotImplementedError(
                    "StatsLogger.add() only supports types: Stats, Dist, and numeric scalars"
                )

    def log(self, step: int):
        """Aggregate buffered values and emit logs.

        Args:
            step: Global step index used for TensorBoard summary steps.
        """
        self._create_logs()
        if self.tensorboard:
            self._log_to_tensorboard(step)
        if self.stdout:
            self._log_to_stdout(step)

    def _create_logs(self):
        """Create :attr:`stats_to_log` and :attr:`dists_to_log` from buffers."""
        self._create_stats_to_log()
        self._create_dists_to_log()

    def _create_stats_to_log(self):
        """Compute smoothed scalar values to emit."""
        self.stats_to_log = {}
        for key, val in self.stats.items():
            if len(val) > 0:
                self.stats_to_log[key] = float(np.mean(val))

    def _create_dists_to_log(self):
        """Collect distributions to emit as histograms."""
        self.dists_to_log = {}
        for key, val in self.dists.items():
            if len(val) > 0:
                self.dists_to_log[key] = val

    def _log_to_tensorboard(self, step: int):
        """Write scalars and histograms to TensorBoard.

        Args:
            step: Global step index used for TensorBoard summary steps.
        """
        assert self.summary_writer is not None, (
            "You're trying to log to tensorboard without a summary writer setup!"
        )
        with self.summary_writer.as_default():
            for key, value in self.stats_to_log.items():
                tf.summary.scalar(self.prefix + key, value, step=step)
            for key, values in self.dists_to_log.items():
                tf.summary.histogram(self.prefix + key, data=values, step=step)

    def _log_to_stdout(self, step: int):
        """Print the current scalar log table to stdout.

        Args:
            step: Global step index (unused; included for API symmetry).
        """
        stats_to_print = {key: "{0:.4g}".format(val) for key, val in self.stats_to_log.items()}
        if not stats_to_print:
            return

        max_key_len = max([len(key) for key in stats_to_print] + [max(0, len(self.prefix) - 2)])
        max_val_len = max([len(val) for val in stats_to_print.values()])

        stdout = ""
        max_len = 1 + 4 + max_key_len + 2 + 1 + 2 + max_val_len + 2 + 1
        stdout += ("-" * max_len + "\n")
        stdout += (
            "|  "
            + self.prefix
            + " " * (2 + max_key_len - len(self.prefix) + 2)
            + "|"
            + " " * (2 + max_val_len + 2)
            + "|\n"
        )
        for key, val in stats_to_print.items():
            stdout += (
                "|    "
                + key
                + " " * (max_key_len - len(key) + 2)
                + "|  "
                + val
                + " " * (max_val_len - len(val) + 2)
                + "|\n"
            )
        stdout += ("-" * max_len + "\n")

        if self.tqdm:
            tqdm.write(stdout)
        else:
            print(stdout)

class RolloutLogger(BaseLogger):
    r"""Logger for episodic metrics produced during environment rollouts.

    This logger is designed for per-episode summaries that arrive via an ``info``
    dict (e.g., from Gymnasium environments). It looks for:

    - ``info["constraint"]["episode"]``: constraint-related episode metrics
    - ``info["metrics"]["episode"]``: generic episode metrics

    and treats the values as scalars.

    It also reports simple runtime diagnostics to stdout:

    - ``fps``: :math:`\frac{\text{timesteps}}{\text{wall-clock seconds}}`
    - ``time_elapsed``: wall-clock seconds since the first :meth:`add`
    - ``total_timesteps``: the provided global step

    Notes:
        The most recent value in each deque is treated as the "current episode"
        and excluded from the mean shown in stdout/TensorBoard (so the displayed
        mean reflects completed episodes only).
    """

    def __init__(
        self,
        stdout: bool = True,
        tqdm: bool = True,
        tensorboard: bool = False,
        summary_writer: Optional[tf.summary.SummaryWriter] = None,
        stats_window_size: int = 100,
        prefix: str = "",
    ):
        super().__init__(
            stdout=stdout,
            tqdm=tqdm,
            tensorboard=tensorboard,
            summary_writer=summary_writer,
            stats_window_size=stats_window_size,
            prefix=prefix,
        )
        self.start_time: Optional[float] = None
        self.stats_to_log: Dict[str, float] = {}

    def add(self, info: Mapping[str, Any], verbose: int = 0):
        """Ingest an ``info`` dict and extract episodic scalars.

        Args:
            info: Rollout ``info`` mapping (typically from environment step).
                If present, the logger reads:
                ``info["constraint"]["episode"]`` and/or
                ``info["metrics"]["episode"]``.
            verbose: Reserved for compatibility; currently unused.
        """
        if self.start_time is None:
            self.start_time = time.time()

        constraint = info.get("constraint", {})
        if isinstance(constraint, Mapping) and "episode" in constraint:
            ep_metrics = constraint["episode"]
            if isinstance(ep_metrics, Mapping):
                self._add_scalars(ep_metrics)

        metrics = info.get("metrics", {})
        if isinstance(metrics, Mapping) and "episode" in metrics:
            ep_metrics = metrics["episode"]
            if isinstance(ep_metrics, Mapping):
                self._add_scalars(ep_metrics)

    def log(self, step: int):
        """Aggregate buffered episode metrics and emit logs.

        Args:
            step: Global step index used for TensorBoard summary steps.
        """
        self._create_logs()
        if self.tensorboard:
            self._log_to_tensorboard(step)
        if self.stdout:
            self._log_to_stdout(step)

    def _create_logs(self):
        """Create :attr:`stats_to_log` from episode buffers."""
        self._create_stats_to_log()

    def _add_scalars(self, scalars: Mapping[str, Union[float, int, np.floating]]):
        """Append scalar episode metrics into rolling windows.

        Args:
            scalars: Mapping from metric names to numeric values.
        """
        for k, v in scalars.items():
            if k in self.stats:
                self.stats[k].append(float(v))
            else:
                # +1 because we keep the most recent value as "current episode".
                self.stats[k] = deque([float(v)], maxlen=self.stats_window_size + 1)
                    
    def _create_stats_to_log(self):
        """Compute per-metric mean over *completed* episodes."""
        self.stats_to_log = {}
        for key, val in self.stats.items():
            if len(val) > 1:
                # Temporarily drop last (current) value from the mean.
                last = val.pop()
                self.stats_to_log[key] = float(np.mean(val)) if len(val) > 0 else float(last)
                val.append(last)

    def _log_to_tensorboard(self, step: int):
        """Write episode scalar summaries to TensorBoard.

        Args:
            step: Global step index used for TensorBoard summary steps.
        """
        assert self.summary_writer is not None, (
            "You're trying to log to tensorboard without a summary writer setup!"
        )
        with self.summary_writer.as_default():
            for key, value in self.stats_to_log.items():
                tf.summary.scalar(self.prefix + key, value, step=step)

    def _log_to_stdout(self, step: int):
        """Print episode summaries and runtime diagnostics to stdout.

        Args:
            step: Global step index used for fps and total timestep display.
        """
        stats_to_print: Dict[str, str] = {
            key: "{0:.4g}".format(val) for key, val in self.stats_to_log.items()
        }
        if self.start_time is not None:
            current_time = time.time()
            elapsed = current_time - self.start_time
            if elapsed > 0:
                stats_to_print["fps"] = "{0:.4g}".format(step / elapsed)
            stats_to_print["time_elapsed"] = "{0:.4g}".format(elapsed)
        stats_to_print["total_timesteps"] = str(step)

        if not stats_to_print:
            return

        max_key_len = max([len(key) for key in stats_to_print] + [max(0, len(self.prefix) - 2)])
        max_val_len = max([len(val) for val in stats_to_print.values()])

        stdout = ""
        max_len = 1 + 4 + max_key_len + 2 + 1 + 2 + max_val_len + 2 + 1
        stdout += ("-" * max_len + "\n")
        stdout += (
            "|  "
            + self.prefix
            + " " * (2 + max_key_len - len(self.prefix) + 2)
            + "|"
            + " " * (2 + max_val_len + 2)
            + "|\n"
        )
        for key, val in stats_to_print.items():
            stdout += (
                "|    "
                + key
                + " " * (max_key_len - len(key) + 2)
                + "|  "
                + val
                + " " * (max_val_len - len(val) + 2)
                + "|\n"
            )
        stdout += ("-" * max_len + "\n")

        if self.tqdm:
            tqdm.write(stdout)
        else:
            print(stdout)

class TrainLogger(BaseLogger):
    r"""Orchestrate multiple loggers for a training run.

    A :class:`TrainLogger` is a thin wrapper around a set of sub-loggers (e.g.
    :class:`StatsLogger`, :class:`RolloutLogger`). It forwards :meth:`add` calls
    to the appropriate sub-logger and aggregates stdout/TensorBoard output.

    Args:
        loggers: A list of ``(name, ctor)`` pairs. ``ctor`` must be a
            :class:`BaseLogger` subclass (or compatible callable) that can be
            constructed with the same keyword arguments as :class:`BaseLogger`.
        stdout: If ``True``, print a combined stdout table for all sub-loggers.
        tqdm: If ``True``, use :func:`tqdm.write` when printing.
        tensorboard: If ``True``, forward TensorBoard logging to sub-loggers.
        summary_writer: TensorBoard writer passed to each sub-logger when
            ``tensorboard=True``.
        stats_window_size: Either a single window size used for all sub-loggers,
            or a list of per-logger window sizes aligned with ``loggers``.
        prefix: Optional display prefix for stdout tables.

    Attributes:
        loggers: Mapping from logger key to instantiated :class:`BaseLogger`.
        start_time: Wall-clock time at which the first :meth:`add` occurred, used
            for runtime diagnostics.
    """

    def __init__(
        self,
        loggers: List[Tuple[str, Any]],
        stdout: bool = True,
        tqdm: bool = True,
        tensorboard: bool = False,
        summary_writer: Optional[tf.summary.SummaryWriter] = None,
        stats_window_size: Union[int, List[int]] = 100,
        prefix: str = "",
    ):
        # Note: TrainLogger is a coordinator and intentionally does not call
        # BaseLogger.__init__ (it does not maintain its own rolling buffers).
        self.loggers: Dict[str, BaseLogger] = {}
        self.stdout: bool = stdout
        self.tqdm: bool = tqdm
        self.tensorboard: bool = tensorboard
        self.summary_writer: Optional[tf.summary.SummaryWriter] = summary_writer
        self.prefix: str = prefix

        if isinstance(stats_window_size, int):
            window_sizes = [stats_window_size] * len(loggers)
        elif isinstance(stats_window_size, list):
            window_sizes = stats_window_size
        else:
            raise RuntimeError("Expected type int or List[int] for stats_window_size")

        if len(window_sizes) != len(loggers):
            raise ValueError("stats_window_size list must match number of loggers")

        for idx, (key, ctor) in enumerate(loggers):
            # ctor is expected to be a BaseLogger subclass or callable returning one.
            self.loggers[key] = ctor(
                stdout=self.stdout,
                tqdm=self.tqdm,
                tensorboard=self.tensorboard,
                summary_writer=self.summary_writer,
                stats_window_size=window_sizes[idx],
                prefix=key,
            )

        self.start_time: Optional[float] = None

    def add(self, key: str, obj: Any):
        """Add an object to a named sub-logger.

        Args:
            key: The sub-logger key as provided in ``loggers`` during
                construction.
            obj: The object to forward to ``self.loggers[key].add(...)``.

        Raises:
            KeyError: If ``key`` is not a configured sub-logger.
        """
        if self.start_time is None:
            self.start_time = time.time()
        self.loggers[key].add(obj)

    def log(self, step: int):
        """Emit TensorBoard logs (per sub-logger) and a combined stdout table.

        Args:
            step: Global step index used for TensorBoard summary steps.
        """
        for key in self.loggers.keys():
            # Rely on sub-logger internal API (common to StatsLogger/RolloutLogger).
            self.loggers[key]._create_logs()  # type: ignore[attr-defined]
            if self.tensorboard:
                self.loggers[key]._log_to_tensorboard(step)  # type: ignore[attr-defined]

        if self.stdout:
            self._log_to_stdout(step)

    def _log_to_stdout(self, step: int) -> None:
        """Print a combined stdout table for all sub-loggers.

        Args:
            step: Global step index used for runtime diagnostics.
        """
        stats_to_print: Dict[str, Dict[str, str]] = {}
        stats_to_print["run"] = {}

        if self.start_time is not None:
            current_time = time.time()
            elapsed = current_time - self.start_time
            if elapsed > 0:
                stats_to_print["run"]["fps"] = "{0:.4g}".format(step / elapsed)
            stats_to_print["run"]["time_elapsed"] = "{0:.4g}".format(elapsed)
        stats_to_print["run"]["total_timesteps"] = str(step)

        for key, logger in self.loggers.items():
            # Sub-loggers populate stats_to_log in _create_logs().
            stats = getattr(logger, "stats_to_log", {})
            stats_to_print[key] = {k: "{0:.4g}".format(v) for k, v in stats.items()}

        max_key_len = 0
        max_val_len = 0
        for group in stats_to_print.values():
            if not group:
                continue
            max_key_len = max(max_key_len, max([len(k) for k in group.keys()] + [max(0, len(self.prefix) - 2)]))
            max_val_len = max(max_val_len, max([len(v) for v in group.values()]))

        stdout = ""
        max_len = 1 + 4 + max_key_len + 2 + 1 + 2 + max_val_len + 2 + 1
        stdout += ("-" * max_len + "\n")
        if self.prefix:
            stdout += (
                "|  "
                + self.prefix
                + " " * (2 + max_key_len - len(self.prefix) + 2)
                + "|"
                + " " * (2 + max_val_len + 2)
                + "|\n"
            )
            stdout += ("-" * max_len + "\n")

        for group_key, group in stats_to_print.items():
            if not group:
                continue
            group_prefix = group_key + "/"
            stdout += (
                "|  "
                + group_prefix
                + " " * (2 + max_key_len - len(group_prefix) + 2)
                + "|"
                + " " * (2 + max_val_len + 2)
                + "|\n"
            )
            for k, v in group.items():
                stdout += (
                    "|    "
                    + k
                    + " " * (max_key_len - len(k) + 2)
                    + "|  "
                    + v
                    + " " * (max_val_len - len(v) + 2)
                    + "|\n"
                )
            stdout += ("-" * max_len + "\n")

        if self.tqdm:
            tqdm.write(stdout)
        else:
            print(stdout)
            

            



            

