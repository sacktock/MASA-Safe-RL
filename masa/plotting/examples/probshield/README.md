# ProbShield plotting example

A worked example of `masa.plotting` for the ProbShield safe RL paper. It registers three diagnostic figure specs on top of the generic `performance` spec that ships with the library:

* `margin` shows margin trajectories per horizon, with one line per seed.
* `margin_dist` shows distribution bands and tail fractions for the logged margin quantile columns.
* `betas` shows internal beta diagnostics for the `cont_v1` variant against the `cont_v2` reference.

## Required metrics

The three example specs read W&B columns that the ProbShield training code logs. If your runs do not log these columns, the corresponding panels will be skipped with a warning.

| Spec | Required columns |
|------|------------------|
| `margin`       | `train/stats/margin_<t>_mean`, `train/stats/margin_<t>_std` |
| `margin_dist`  | `train/stats/margin_<t>_q05` / `_q25` / `_q50` / `_q75` / `_q95`, `train/stats/margin_frac_near_0`, `train/stats/margin_frac_near_1` |
| `betas`        | `train/stats/betas_std`, `train/stats/mix_std`, `train/stats/betas_min`, `train/stats/betas_max`, `train/stats/proj_penalty`, `train/stats/betas_mean`, `train/stats/mix_mean` |

The generic `performance` spec only needs `eval/rollout/ep_reward` and `eval/rollout/satisfied`.

## Usage

```bash
python -m masa.plotting \
    --config masa/plotting/examples/probshield/paper.yaml \
    --specs-from masa.plotting.examples.probshield \
    --stages all
```

The `--specs-from` flag loads this folder's `__init__.py`, which triggers the three `register(PlotSpec(...))` calls. Without that flag the four registered figures collapse to just `performance`.

Render a subset:

```bash
python -m masa.plotting \
    --config masa/plotting/examples/probshield/paper.yaml \
    --specs-from masa.plotting.examples.probshield \
    --specs performance,margin
```

Dry run prints the resolved plan with no W&B calls or disk writes:

```bash
python -m masa.plotting \
    --config masa/plotting/examples/probshield/paper.yaml \
    --specs-from masa.plotting.examples.probshield \
    --stages all --dry-run
```

## Configuration

`paper.yaml` declares the four ProbShield variants (`disc_v1`, `disc_v2`, `cont_v1`, `cont_v2`) with the colourblind palette from the paper. Edit it freely. The `betas` spec refers to `cont_v1` and `cont_v2` by name, so if you rename those variants you will also need to update [betas.py](betas.py).

`cache_dir: ./wandb_cache` and `output_dir: ./plots` resolve relative to the YAML, so the cache and rendered figures land inside this example folder.
