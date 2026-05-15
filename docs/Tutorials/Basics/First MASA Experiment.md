# First MASA Experiment

This tutorial gets MASA running end to end with a tiny safe RL experiment. You will:

- preview the `bridge_crossing` environment,
- wrap it with `make_env`,
- inspect labels and constraint metrics,
- run a tiny `q_learning` smoke experiment,
- read the training and evaluation logs.

Runnable notebook: [notebooks/tutorials/01_first_masa_experiment.ipynb](../../../notebooks/tutorials/01_first_masa_experiment.ipynb)

## CPU-First Setup

Use CPU for this first run. This keeps the tutorial portable and avoids noisy CUDA probing on machines that have GPU packages installed.

```python
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
```

Set those environment variables before importing MASA/JAX modules.

## Preview Bridge Crossing

`bridge_crossing` is a tabular gridworld. The agent starts at the lower-left corner, receives reward for reaching the goal row, and incurs safety cost when it reaches lava.

```python
from IPython.display import display
from PIL import Image

from masa.envs.tabular.bridge_crossing import BridgeCrossing

preview_env = BridgeCrossing(render_mode="rgb_array", render_window_size=320)
obs, info = preview_env.reset(seed=0)
print({"reset_obs": obs, "reset_info": info})
display(Image.fromarray(preview_env.render()))
preview_env.close()
```

## Build a MASA Environment

`make_env` applies MASA's standard wrapper stack:

```text
TimeLimit -> LabelledEnv -> ConstraintEnv -> ConstraintMonitor -> RewardMonitor
```

For this first experiment, use the PCTL constraint with the default Bridge Crossing label and cost functions.

```python
from pprint import pprint

from masa.plugins.helpers import load_plugins
from masa.common.utils import make_env
from masa.envs.tabular.bridge_crossing import cost_fn, label_fn

load_plugins()

def build_masa_env():
    return make_env(
        "bridge_crossing",
        "pctl",
        400,
        label_fn=label_fn,
        cost_fn=cost_fn,
        alpha=0.01,
    )

env = build_masa_env()
obs, info = env.reset(seed=0)

print("reset observation:", obs)
print('info["labels"]:', info["labels"])
print('info["constraint"]:')
pprint(info["constraint"])
```

The observation and reward still follow Gymnasium. MASA adds semantic safety information through `info["labels"]` and `info["constraint"]`.

## Step by Hand

Step a few fixed actions before training. This is the fastest way to confirm what your environment emits.

```python
ACTION_NAMES = {0: "left", 1: "right", 2: "down", 3: "up", 4: "stay"}
scripted_actions = [3, 3, 3, 4, 1]
rows = []

for step, action in enumerate(scripted_actions, start=1):
    obs, reward, terminated, truncated, info = env.step(action)
    rows.append(
        {
            "step": step,
            "action": ACTION_NAMES[action],
            "obs": int(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "labels": sorted(info["labels"]),
            "constraint_step": info["constraint"]["step"],
        }
    )
    if terminated or truncated:
        break

pprint(rows)
env.close()
```

## Train a Tiny Q-Learner

This smoke run is deliberately tiny. It is not meant to learn a good policy; it proves that the environment, constraint, algorithm, evaluation, and logger all connect correctly.

```python
from masa.algorithms.tabular import QL

train_env = build_masa_env()
eval_env = build_masa_env()

algo = QL(
    train_env,
    tensorboard_logdir=None,
    seed=0,
    monitor=True,
    device="cpu",
    verbose=0,
    env_fn=build_masa_env,
    eval_env=eval_env,
)

algo.train(
    num_frames=20,
    eval_freq=10,
    log_freq=10,
    num_eval_episodes=1,
    stats_window_size=10,
)

train_env.close()
eval_env.close()
```

You should see log groups like:

- `train/rollout`: episode-level constraint metrics from training.
- `train/stats`: algorithm statistics, such as Q-learning step size and exploration temperature.
- `eval/rollout`: evaluation metrics, including constraint satisfaction, reward, and episode length.

## CLI Equivalent

The same smoke run can be launched from a shell:

```sh
JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 uv run --locked python -m masa.run \
  --custom-cfgs bridge_crossing \
  --algo q_learning \
  --total-timesteps 20 \
  --seed 0 \
  --run.eval_every 10 \
  --run.log_every 10 \
  --run.eval_episodes 1 \
  --run.device cpu
```

For a real experiment, increase `num_frames` or `--total-timesteps`, run multiple seeds, and keep this short tutorial run as a quick correctness check.
