# Plotting

This module provides a **unified Seaborn plotting interface for TensorBoard and Weights & Biases results**. Metrics are discovered dynamically from whichever backend was used during training, so no hardcoded metric lists are required.

## Usage

### 1. Programmatic (after training)

Call `plot_run()` directly after `algo.train()` using the variables already in scope:

```python
from masa.plotting.api import plot_run

algo.train(num_frames=500_000, ...)

# Plot from W&B
if algo.wandb_project:
    plot_run(
        wandb_project=algo.wandb_project,
        run_filter=algo.wandb_name,
        metrics=["train/rollout/ep_reward", "train/rollout/satisfied"],
        output_path="images/mini_pacman/results.pdf",
    )

# Plot from TensorBoard
elif algo.tensorboard_logdir:
    plot_run(
        tensorboard_logdir=algo.tensorboard_logdir,
        metrics=["train/rollout/ep_reward", "train/rollout/satisfied"],
        output_path="images/mini_pacman/results.pdf",
    )
```

### 2. CLI

Plot from W&B:

```bash
python masa/plotting/run.py --wandb ProbShield-Benchmarks -m train/rollout/ep_reward train/rollout/satisfied -o results.pdf
```

Plot from TensorBoard:

```bash
python masa/plotting/run.py --tensorboard logdir/mini_pacman -m train/rollout/ep_reward -o results.pdf
```

List all available metrics without plotting:

```bash
python masa/plotting/run.py --wandb ProbShield-Benchmarks --list-metrics
```

Filter runs by name substring:

```bash
python masa/plotting/run.py --wandb ProbShield-Benchmarks --run-filter mini_pacman_cont_v2 -o filtered.pdf
```

### 3. Direct DataFrame Plotting

For advanced use, load data into a DataFrame and plot manually:

```python
from masa.plotting.sources import WandBSource
from masa.plotting.plot import plot_metrics

source = WandBSource("ProbShield-Benchmarks")
df = source.load(metrics=["train/rollout/ep_reward"])

fig = plot_metrics(df, smooth_weight=0.8, output_path="custom_plot.pdf")
```

The same interface works with TensorBoard:

```python
from masa.plotting.sources import TensorBoardSource

source = TensorBoardSource("logdir/mini_pacman")
print(source.list_metrics())  # discover available metrics
df = source.load(metrics=["train/rollout/ep_reward", "train/rollout/satisfied"])
```

Both sources return a normalised DataFrame with columns `[step, metric, value, run]`.

## Module Structure

| File         | Role                                                  |
| ------------ | ----------------------------------------------------- |
| `api.py`     | `plot_run()` high-level entry point                   |
| `sources.py` | `TensorBoardSource` and `WandBSource` data loaders    |
| `plot.py`    | `plot_metrics()` Seaborn rendering with EMA smoothing |
| `run.py`     | CLI entry point                                       |
