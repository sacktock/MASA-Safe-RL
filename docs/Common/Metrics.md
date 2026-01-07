# Metrics

## Overview

```{eval-rst}
MASA metrics are designed to make it easy to record and report learning signals at different granularities: single scalars, streaming summary statistics, and approximate distributions. Keeping logging lightweight and backend-agnostic.

- **Scalars** are plain numeric values (e.g., reward, loss, episode length). Loggers keep a rolling window of recent scalar values and typically report a smoothed mean over that window.
- **Summary statistics** are handled by :class:`masa.common.metrics.Stats`, which performs *streaming aggregation* over batches of values. It tracks running moments (mean and mean-square) and extrema (min/max/magnitude), and exposes derived quantities like standard deviation:
  :math:`\sigma = \sqrt{\max(0, \mathbb{E}[X^2] - \mathbb{E}[X]^2)}`.
  Calling :meth:`Stats.get` returns a flat dictionary of scalars suitable for logging.
- **Distributions** are handled by :class:`masa.common.metrics.Dist`, which maintains a fixed-size *reservoir sample* of a stream. This gives a compact approximation of the underlying distribution that can be plotted or logged as a histogram.
- **Logging** is performed by logger implementations (see the next page), which accept mixtures of scalars, :class:`~masa.common.metrics.Stats`, and :class:`~masa.common.metrics.Dist` objects. They aggregate values over a configurable window and can emit to stdout and/or TensorBoard with consistent key prefixing.
```

## Metrics Core

```{eval-rst}
.. autoclass:: masa.common.metrics.Stats
    :members:
    :show-inheritance:
.. autoclass:: masa.common.metrics.Dist
    :members:
    :show-inheritance:
```

## Next Steps

- **[Logging](Metrics/Logging)**

```{toctree}
:caption: Metrics
:hidden:

Metrics/Logging
```