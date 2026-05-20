# masa.plotting

Plotting utilities for MASA training and evaluation runs. Two interfaces live in this package:

* **Single run plotter** (`masa.plotting.api.plot_run`, `masa/plotting/run.py`): tidy seaborn line charts from one TensorBoard log directory or one W&B project. Use this when you have just trained an algorithm and want a quick visual of its metrics.
* **Benchmark pipeline** (`python -m masa.plotting`): download many W&B runs, aggregate quantiles across seeds, and render a registry of declarative figure specs. Use this for paper grade plots that compare variants across environments.

Both interfaces share the data sources in [sources.py](sources.py). The benchmark pipeline adds a `CachedWandbSource` for per run CSV caching with bounded retries.

## Installation

All required dependencies are declared in the project's `pyproject.toml`:

```bash
uv sync
```

## Single run plotter

From Python, after a training run:

```python
from masa.plotting.api import plot_run

plot_run(
    wandb_project="your-project-name",
    run_filter="my_env_baseline_v1_seed1",
    metrics=["train/rollout/ep_reward", "train/rollout/satisfied"],
    output_path="images/results.pdf",
)
```

Or from a shell:

```bash
python masa/plotting/run.py --wandb your-project-name \
    -m train/rollout/ep_reward train/rollout/satisfied \
    -o results.pdf
```

List every metric available in a project before plotting:

```bash
python masa/plotting/run.py --wandb your-project-name --list-metrics
```

The same interface accepts `--tensorboard <logdir>` instead of `--wandb`.

## Benchmark pipeline

The pipeline ships with two generic specs: `performance`, which uses the standard MASA evaluation metrics (`eval/rollout/ep_reward` and `eval/rollout/satisfied`), and `margin`, which draws per-horizon fan charts of the shield's safety budget from `train/stats/margin_<t>` metrics. Run them against your W&B project:

```bash
python -m masa.plotting --config masa/plotting/configs/example.yaml --stages all
```

`example.yaml` carries placeholder values. Copy it, fill in your W&B entity and project, list your variants, then point `--config` at your copy.

Dry run prints the resolved plan with no W&B calls and no disk writes:

```bash
python -m masa.plotting --config <your-yaml> --stages all --dry-run
```

Render only a subset of registered figures:

```bash
python -m masa.plotting --config <your-yaml> --specs performance
```

Bust the download cache (the long-form and quantile frames are always rebuilt in memory, so only the per-run W&B cache needs busting):

```text
--force-download    refetch every run from W&B
```

Load extra figure specs from another module (the package ships `performance` and `margin` by default):

```bash
python -m masa.plotting --config <your-yaml> --specs-from <dotted.module>
```

Use this flag to register your own `PlotSpec` modules alongside the default `performance` and `margin` specs.

## Configuration

Annotated schema, taken from [configs/example.yaml](configs/example.yaml):

```yaml
wandb_entity: your-wandb-entity
wandb_project: your-project-name

# Regex with three named groups (env, variant, seed) applied to every run name.
run_schema:
  pattern: '^(?P<env>.+?)_(?P<variant>[a-z_]+_v\d+)_seed(?P<seed>\d+)$'

# Paths are resolved relative to the YAML file's parent directory.
cache_dir: ./wandb_cache
output_dir: ./plots

palette: colorblind
seed_for_logged_quantiles: 1
metrics: null   # null fetches every non internal column
envs: null      # null renders every env found in the data; or list a subset

variants:
  - {name: baseline,  label: 'Baseline',  colour: '#0072B2', order: 0}
  - {name: my_method, label: 'My Method', colour: '#D55E00', order: 1}
```

Adding or removing a variant is a YAML edit, no Python changes. Leaving `colour` unset auto assigns a value from the named seaborn palette. Runs whose names fail to match `run_schema.pattern` are logged and skipped.

## How the pipeline works

```
example.yaml
   |
   v
Config (config.py)
   |
   +--> download   (sources.py :: CachedWandbSource)
   |       writes per run CSVs into cache_dir
   |
   +--> process    (processing.py)
   |       reads per run CSVs
   |       builds long form : [env, variant, seed, step, metric, value]
   |       builds quantiles : [env, variant, step, metric, q05, q25, q50, q75, q95]
   |
   +--> render     (render.py + specs/*)
           for each registered PlotSpec, for each env:
               build a RenderContext
               resolve panels (static tuple or callable)
               draw each panel by dispatching on Panel.source
               save the figure to output_dir/<spec.out_subdir>/<spec.id>_<env>.png
```

### Stage 1: load config

`config.load_config(path)` parses the YAML into a frozen `Config` dataclass. Any variant whose `colour` is omitted is filled from the named seaborn palette. Relative paths resolve against the YAML file's parent directory.

### Stage 2: download

`CachedWandbSource.download()` calls `wandb.Api().runs(entity/project)` with tenacity retries (three attempts, exponential backoff). For each run, it writes the full history to `cache_dir/{run.name}.csv` via an atomic rename. If the file already exists and `force_download` is false, the API call is skipped. Failures on a single run are logged and the loop continues, so one bad run never breaks the batch.

### Stage 3: process

`processing.build_long_form(config)` reads every CSV in `cache_dir`, matches the filename against `run_schema.pattern` to recover `(env, variant, seed)`, melts every metric column into rows, coerces each cell to a scalar (stringified `wandb.Histogram` payloads are collapsed to their median), drops NaNs, and concatenates the result. The output frame uses the canonical long form schema `[env, variant, seed, step, metric, value]`.

`processing.build_quantile_frame(long)` groups that long form by `(env, variant, step, metric)` and computes five quantiles (`q05`, `q25`, `q50`, `q75`, `q95`) across seeds, feeding every `aggregated` panel.

Both frames are rebuilt in memory on every run. Only the per-run W&B download is cached on disk (`cache_dir/{run.name}.csv`); the in-memory melt and groupby are cheap enough that a persistent frame cache isn't worth the staleness risk.

### Stage 4: render

`Pipeline.render()` walks the registered `PlotSpec` objects. For each spec and each env it constructs a `RenderContext` (long frame, quantile frame, variants, env metadata) and calls `render.render_spec(spec, ctx, output_dir)`.

The renderer resolves the spec's panels (a static tuple, or a callable that consults `ctx.long_df` to discover horizons or variants), resolves the grid, creates the figure, and dispatches each `Panel` by its `source` field:

| `Panel.source`    | What it draws                                                                | Data source     |
|-------------------|------------------------------------------------------------------------------|-----------------|
| `aggregated`      | Central line plus a shaded band between two quantile columns                 | quantile frame  |
| `per_seed`        | One line per seed (solid for seed 1, dashed for later seeds)                 | long form frame |
| `logged_quantile` | Central line plus bands from columns already logged by training, seed 1 only | long form frame |

Two escape hatches handle the cases that resist a uniform declarative description:

* `Panel.custom` lets a panel draw itself with full access to the axes and the render context. Use this when one panel needs to plot different metrics per variant.
* `PlotSpec.figure_builder` bypasses the uniform grid entirely and lets a spec build its own figure via `matplotlib.gridspec`. Use this when the layout has spanning rows or columns.

### Stage 5: save

Each figure is saved as `output_dir/<spec.out_subdir>/<spec.id>_<env>.png` at 200 dpi. If a spec's panels resolve to nothing or its `figure_builder` draws nothing (for example because `long_df` is empty for the requested env), the figure is dropped and a log line records the skip.

## Adding a new spec

Create a Python file (anywhere on the import path), declare a `PlotSpec`, register it:

```python
# my_specs.py
from masa.plotting.specs import register
from masa.plotting.specs.base import Band, Panel, PlotSpec

my_plot = register(PlotSpec(
    id="my_plot",
    grid=(1, 2),
    panels=(
        Panel(metric="eval/rollout/ep_reward", title="Reward",
              bands=(Band("q25", "q75"),)),
        Panel(metric="eval/rollout/satisfied", title="Satisfied",
              bands=(Band("q25", "q75"),)),
    ),
    out_subdir="my_plot",
))
```

Load it via the CLI:

```bash
python -m masa.plotting --config <your-yaml> --specs-from my_specs
```

The new spec is then available via `--specs my_plot`.

## File map

| File                       | Role                                                          |
|----------------------------|---------------------------------------------------------------|
| `__main__.py`              | Benchmark pipeline CLI                                        |
| `api.py`                   | `plot_run` single run interface                               |
| `run.py`                   | Single run CLI                                                |
| `plot.py`                  | `plot_metrics` seaborn renderer                               |
| `sources.py`               | `TensorBoardSource`, `WandBSource`, `CachedWandbSource`       |
| `config.py`                | `Config`, `VariantStyle`, `RunSchema`, `load_config`          |
| `processing.py`            | Long form and quantile frame builders                         |
| `render.py`                | Generic spec renderer                                         |
| `pipeline.py`              | `Pipeline` orchestrator                                       |
| `io_utils.py`              | Logging setup, atomic CSV writes                              |
| `configs/example.yaml`     | Annotated template config                                     |
| `specs/__init__.py`        | Spec registry (auto registers `performance`)                  |
| `specs/base.py`            | `PlotSpec`, `Panel`, `Band`, `HLine`, `RenderContext`         |
| `specs/performance.py`     | Generic 2 by 2 reward and satisfaction figure                 |
