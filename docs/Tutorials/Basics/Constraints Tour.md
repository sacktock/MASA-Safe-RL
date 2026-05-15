# Constraints Tour

This tutorial compares MASA's registered single-agent constraints on the same environment and action scripts. The point is not to train an agent; it is to see how the same labels can produce different safety metrics.

Runnable notebook: [notebooks/tutorials/04_constraints_tour.ipynb](../../../notebooks/tutorials/04_constraints_tour.ipynb)

## Setup

Use CPU-first setup before importing MASA/JAX modules:

```python
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
```

Import the shared environment, the central factory, and the small LTL helpers:

```python
from pprint import pprint

from masa.plugins.helpers import load_plugins
from masa.common.ltl import Atom, DFA
from masa.common.utils import make_env
from masa.envs.tabular.colour_grid_world import cost_fn, label_fn

load_plugins()
```

## Shared Environment and Scripts

All examples use `colour_grid_world`. The blue state is unsafe for cost-based constraints, and the goal state is the reach target for reach-avoid.

```python
UNSAFE_SCRIPT = {"seed": 1, "actions": [2, 2, 2, 2]}
GOAL_SCRIPT = {"seed": 4, "actions": [2] * 8 + [1] * 8}

def make_never_blue_dfa():
    dfa = DFA([0, 1], 0, [1])
    dfa.add_edge(0, 1, Atom("blue"))
    return dfa
```

The unsafe script reaches `blue`; the goal script reaches `goal` without visiting `blue`.

## Visual Overview

The notebook draws these diagrams with small helper functions. The static docs use matching rendered SVG assets, so the page stays readable without exposing the drawing code.

### Seeded Trace Maps

Unsafe script:

```{figure} ../../_static/tutorials/constraints_tour/unsafe_trace.svg
:alt: Unsafe script reaches blue in colour_grid_world.
:width: 446px

Seed `1` with actions `[2, 2, 2, 2]` reaches `blue`.
```

Goal script:

```{figure} ../../_static/tutorials/constraints_tour/goal_trace.svg
:alt: Goal script reaches goal without visiting blue in colour_grid_world.
:width: 446px

Seed `4` with actions `[2] * 8 + [1] * 8` reaches `goal` without visiting `blue`.
```

### CMDP and Probabilistic Safety

`cmdp` and `prob` both start from the same label-derived unsafe signal. They differ in how they aggregate it over an episode.

```{figure} ../../_static/tutorials/constraints_tour/cmdp_prob.svg
:alt: CMDP and probabilistic safety both derive metrics from a label-based unsafe signal.
:width: 900px

`cmdp` checks accumulated cost against a budget, while `prob` checks the fraction of unsafe steps against `alpha`.
```

### Constraint Semantics

The same labels are fed to every constraint. The difference is the safety state each constraint derives from those labels.

```{figure} ../../_static/tutorials/constraints_tour/constraint_semantics.svg
:alt: Same labels feeding different constraint semantics.
:width: 760px

`cmdp`, `prob`, `pctl`, `reach_avoid`, and `ltl_safety` interpret the shared labels differently.
```

## Build Each Constraint

The base environment and labels stay fixed. Only the constraint wrapper and its configuration change.

```python
CONSTRAINT_NAMES = ["cmdp", "prob", "pctl", "reach_avoid", "ltl_safety"]

def build_constraint_env(name, max_episode_steps=40):
    if name == "cmdp":
        return make_env(
            "colour_grid_world",
            "cmdp",
            max_episode_steps,
            label_fn=label_fn,
            cost_fn=cost_fn,
            budget=0.0,
        )
    if name == "prob":
        return make_env(
            "colour_grid_world",
            "prob",
            max_episode_steps,
            label_fn=label_fn,
            cost_fn=cost_fn,
            alpha=0.1,
        )
    if name == "pctl":
        return make_env(
            "colour_grid_world",
            "pctl",
            max_episode_steps,
            label_fn=label_fn,
            cost_fn=cost_fn,
            alpha=0.01,
        )
    if name == "reach_avoid":
        return make_env(
            "colour_grid_world",
            "reach_avoid",
            max_episode_steps,
            label_fn=label_fn,
            avoid_label="blue",
            reach_label="goal",
        )
    if name == "ltl_safety":
        return make_env(
            "colour_grid_world",
            "ltl_safety",
            max_episode_steps,
            label_fn=label_fn,
            dfa=make_never_blue_dfa(),
            obs_type="dict",
        )
    raise ValueError(f"unknown constraint: {name}")
```

## Run the Same Script Through Each Constraint

The helper below records the final row for each constraint. `ltl_safety` augments observations with automaton state, so the observation simplifier handles dictionaries as well as integers.

```python
ACTION_NAMES = {0: "left", 1: "right", 2: "down", 3: "up", 4: "stay"}

def simplify_obs(obs):
    if isinstance(obs, dict):
        return {key: simplify_obs(value) for key, value in obs.items()}
    try:
        return int(obs)
    except (TypeError, ValueError):
        return obs

def run_constraint(name, *, seed, actions, max_episode_steps=40):
    env = build_constraint_env(name, max_episode_steps=max_episode_steps)
    obs, info = env.reset(seed=seed)
    rows = [
        {
            "event": "reset",
            "obs": simplify_obs(obs),
            "labels": sorted(info["labels"]),
            "constraint": info["constraint"],
            "automaton_state": info.get("automaton_state"),
        }
    ]

    for step, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "event": f"step_{step}",
                "action": ACTION_NAMES[action],
                "obs": simplify_obs(obs),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "labels": sorted(info["labels"]),
                "constraint": info["constraint"],
                "automaton_state": info.get("automaton_state"),
            }
        )
        if terminated or truncated:
            break

    env.close()
    return rows

def final_metrics_for(script):
    return {
        name: run_constraint(name, seed=script["seed"], actions=script["actions"])[-1]
        for name in CONSTRAINT_NAMES
    }
```

## Unsafe Script: Reach Blue

With seed `1`, actions `[2, 2, 2, 2]` reach the blue state. Compare how each constraint reports the same labelled event.

```python
unsafe_results = final_metrics_for(UNSAFE_SCRIPT)
pprint(unsafe_results)
```

Things to notice:

- `cmdp` reports `cost=1.0`, `violation=1.0`, and cumulative cost above the zero budget.
- `prob` reports an unsafe fraction above `alpha=0.1`.
- `pctl` reports the PCTL-style condition as not satisfied for the unsafe trace.
- `reach_avoid` reports `violated=True`.
- `ltl_safety` moves the DFA to the accepting unsafe state.

## Goal Script: Reach Goal Without Blue

With seed `4`, actions `[2] * 8 + [1] * 8` reach the goal state without visiting blue.

```python
goal_results = final_metrics_for(GOAL_SCRIPT)
pprint(goal_results)
```

Things to notice:

- cost-based constraints remain satisfied because no blue state was visited,
- `reach_avoid` is satisfied because `goal` was reached before `blue`,
- `ltl_safety` is satisfied because the never-blue DFA never entered its accepting unsafe state.

## How to Read the Differences

- `cmdp` accumulates scalar cost and checks it against a budget.
- `prob` tracks the empirical fraction of unsafe observations and checks it against `alpha`.
- `pctl` is intended to evaluate whether a bounded PCTL formula's satisfaction probability meets its threshold.
- `reach_avoid` separately tracks whether the target was reached and whether the avoid label was ever seen.
- `ltl_safety` advances a DFA and reports violations when the automaton enters an accepting unsafe state.

The raw environment labels are the same. The constraint determines how those labels become safety state and metrics.
