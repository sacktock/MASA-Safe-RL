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
from masa.envs.tabular.colour_grid_world import (
    BLUE_STATE,
    GOAL_STATE,
    GREEN_STATE,
    GRID_SIZE,
    PURPLE_STATE,
    START_STATE,
    cost_fn,
    label_fn,
)

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

The notebook draws these diagrams with small inline SVG helpers. The trace maps below show the real states produced by the seeded scripts.

### Seeded Trace Maps

Unsafe script:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 446 492" width="446" height="492" role="img" aria-label="Unsafe script reaches blue">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="34" y="24" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">Unsafe script: blue is reached</text>
  <rect x="34" y="34" width="378" height="378" fill="#f9fafb" stroke="#d1d5db"/>
  <rect x="34" y="34" width="42" height="42" fill="#dbeafe"/>
  <rect x="202" y="34" width="42" height="42" fill="#ddd6fe"/>
  <rect x="34" y="202" width="42" height="42" fill="#93c5fd"/>
  <rect x="202" y="202" width="42" height="42" fill="#bbf7d0"/>
  <rect x="370" y="370" width="42" height="42" fill="#dcfce7"/>
  <path d="M76 34V412 M118 34V412 M160 34V412 M202 34V412 M244 34V412 M286 34V412 M328 34V412 M370 34V412 M34 76H412 M34 118H412 M34 160H412 M34 202H412 M34 244H412 M34 286H412 M34 328H412 M34 370H412" stroke="#d1d5db" stroke-width="1"/>
  <text x="38" y="70" font-family="sans-serif" font-size="9" fill="#374151">start</text>
  <text x="206" y="70" font-family="sans-serif" font-size="9" fill="#374151">purple</text>
  <text x="38" y="238" font-family="sans-serif" font-size="9" fill="#374151">blue</text>
  <text x="206" y="238" font-family="sans-serif" font-size="9" fill="#374151">green</text>
  <text x="374" y="406" font-family="sans-serif" font-size="9" fill="#374151">goal</text>
  <polyline points="55,55 55,97 55,139 55,181 55,223" fill="none" stroke="#111827" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" opacity="0.72"/>
  <g font-family="sans-serif" font-size="11" font-weight="700" fill="#ffffff" text-anchor="middle">
    <circle cx="55" cy="55" r="10" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="55" y="59">0</text>
    <circle cx="55" cy="97" r="10" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="55" y="101">1</text>
    <circle cx="55" cy="139" r="10" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="55" y="143">2</text>
    <circle cx="55" cy="181" r="10" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="55" y="185">3</text>
    <circle cx="55" cy="223" r="12" fill="#b91c1c" stroke="#ffffff" stroke-width="2"/><text x="55" y="227">4</text>
  </g>
  <text x="34" y="440" font-family="sans-serif" font-size="12" fill="#374151">Numbers are reset/step indices from the actual seeded rollout.</text>
</svg>

Goal script:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 446 492" width="446" height="492" role="img" aria-label="Goal script reaches goal without blue">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="34" y="24" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">Goal script: goal is reached without blue</text>
  <rect x="34" y="34" width="378" height="378" fill="#f9fafb" stroke="#d1d5db"/>
  <rect x="34" y="34" width="42" height="42" fill="#dbeafe"/>
  <rect x="202" y="34" width="42" height="42" fill="#ddd6fe"/>
  <rect x="34" y="202" width="42" height="42" fill="#93c5fd"/>
  <rect x="202" y="202" width="42" height="42" fill="#bbf7d0"/>
  <rect x="370" y="370" width="42" height="42" fill="#dcfce7"/>
  <path d="M76 34V412 M118 34V412 M160 34V412 M202 34V412 M244 34V412 M286 34V412 M328 34V412 M370 34V412 M34 76H412 M34 118H412 M34 160H412 M34 202H412 M34 244H412 M34 286H412 M34 328H412 M34 370H412" stroke="#d1d5db" stroke-width="1"/>
  <text x="38" y="70" font-family="sans-serif" font-size="9" fill="#374151">start</text>
  <text x="206" y="70" font-family="sans-serif" font-size="9" fill="#374151">purple</text>
  <text x="38" y="238" font-family="sans-serif" font-size="9" fill="#374151">blue</text>
  <text x="206" y="238" font-family="sans-serif" font-size="9" fill="#374151">green</text>
  <text x="374" y="406" font-family="sans-serif" font-size="9" fill="#374151">goal</text>
  <polyline points="55,55 55,97 55,139 55,181 97,181 97,223 97,265 97,307 97,349 139,349 181,349 223,349 265,349 307,349 349,349 349,391 391,391" fill="none" stroke="#111827" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" opacity="0.72"/>
  <g font-family="sans-serif" font-size="9" font-weight="700" fill="#ffffff" text-anchor="middle">
    <circle cx="55" cy="55" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="55" y="58">0</text>
    <circle cx="55" cy="97" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="55" y="100">1</text>
    <circle cx="55" cy="139" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="55" y="142">2</text>
    <circle cx="55" cy="181" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="55" y="184">3</text>
    <circle cx="97" cy="181" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="97" y="184">4</text>
    <circle cx="97" cy="223" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="97" y="226">5</text>
    <circle cx="97" cy="265" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="97" y="268">6</text>
    <circle cx="97" cy="307" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="97" y="310">7</text>
    <circle cx="97" cy="349" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="97" y="352">8</text>
    <circle cx="139" cy="349" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="139" y="352">9</text>
    <circle cx="181" cy="349" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="181" y="352">10</text>
    <circle cx="223" cy="349" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="223" y="352">11</text>
    <circle cx="265" cy="349" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="265" y="352">12</text>
    <circle cx="307" cy="349" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="307" y="352">13</text>
    <circle cx="349" cy="349" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="349" y="352">14</text>
    <circle cx="349" cy="391" r="9" fill="#111827" stroke="#ffffff" stroke-width="2"/><text x="349" y="394">15</text>
    <circle cx="391" cy="391" r="12" fill="#166534" stroke="#ffffff" stroke-width="2"/><text x="391" y="394">16</text>
  </g>
  <text x="34" y="440" font-family="sans-serif" font-size="12" fill="#374151">Numbers are reset/step indices from the actual seeded rollout.</text>
</svg>

### DFA and Constraint Semantics

The `ltl_safety` example uses a two-state DFA. The accepting state is unsafe here: entering it means the safety property has been violated.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 190" width="520" height="190" role="img" aria-label="Never blue DFA">
  <defs>
    <marker id="docs-dfa-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#111827" />
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="28" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">Never-blue DFA used by ltl_safety</text>
  <line x1="55" y1="96" x2="91" y2="96" stroke="#111827" stroke-width="2.5" marker-end="url(#docs-dfa-arrow)"/>
  <circle cx="140" cy="96" r="40" fill="#dcfce7" stroke="#166534" stroke-width="3"/>
  <text x="140" y="91" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#14532d">q0</text>
  <text x="140" y="110" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#14532d">safe</text>
  <circle cx="380" cy="96" r="42" fill="#fee2e2" stroke="#991b1b" stroke-width="3"/>
  <circle cx="380" cy="96" r="34" fill="none" stroke="#991b1b" stroke-width="2"/>
  <text x="380" y="91" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#7f1d1d">q1</text>
  <text x="380" y="110" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#7f1d1d">unsafe</text>
  <path d="M180 96 C235 52 285 52 338 96" fill="none" stroke="#111827" stroke-width="2.5" marker-end="url(#docs-dfa-arrow)"/>
  <text x="260" y="55" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">blue</text>
  <path d="M122 58 C86 25 190 25 158 58" fill="none" stroke="#166534" stroke-width="2" marker-end="url(#docs-dfa-arrow)"/>
  <text x="140" y="24" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#166534">implicit not blue loop</text>
  <path d="M362 58 C326 25 430 25 398 58" fill="none" stroke="#991b1b" stroke-width="2" marker-end="url(#docs-dfa-arrow)"/>
  <text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#991b1b">implicit loop after violation</text>
</svg>

The same labels are fed to every constraint. The difference is the safety state each constraint derives from those labels.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 310" width="760" height="310" role="img" aria-label="Constraint semantics diagram">
  <defs><marker id="docs-sem-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#4b5563" /></marker></defs>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="30" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">Same labels, different constraint semantics</text>
  <rect x="26" y="86" width="160" height="88" rx="8" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="106" y="117" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">labels</text>
  <text x="106" y="140" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">blue, goal, ...</text>
  <line x1="186" y1="130" x2="270" y2="78" stroke="#4b5563" stroke-width="1.8" marker-end="url(#docs-sem-arrow)"/>
  <rect x="280" y="58" width="430" height="36" rx="7" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="300" y="81" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2937">cmdp</text>
  <text x="420" y="81" font-family="sans-serif" font-size="12" fill="#374151">sum cost &lt;= budget</text>
  <line x1="186" y1="130" x2="270" y2="124" stroke="#4b5563" stroke-width="1.8" marker-end="url(#docs-sem-arrow)"/>
  <rect x="280" y="104" width="430" height="36" rx="7" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="300" y="127" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2937">prob</text>
  <text x="420" y="127" font-family="sans-serif" font-size="12" fill="#374151">unsafe fraction &lt;= alpha</text>
  <line x1="186" y1="130" x2="270" y2="170" stroke="#4b5563" stroke-width="1.8" marker-end="url(#docs-sem-arrow)"/>
  <rect x="280" y="150" width="430" height="36" rx="7" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="300" y="173" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2937">pctl</text>
  <text x="420" y="173" font-family="sans-serif" font-size="12" fill="#374151">safety satisfied so far</text>
  <line x1="186" y1="130" x2="270" y2="216" stroke="#4b5563" stroke-width="1.8" marker-end="url(#docs-sem-arrow)"/>
  <rect x="280" y="196" width="430" height="36" rx="7" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="300" y="219" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2937">reach_avoid</text>
  <text x="420" y="219" font-family="sans-serif" font-size="12" fill="#374151">reach goal before blue</text>
  <line x1="186" y1="130" x2="270" y2="262" stroke="#4b5563" stroke-width="1.8" marker-end="url(#docs-sem-arrow)"/>
  <rect x="280" y="242" width="430" height="36" rx="7" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="300" y="265" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f2937">ltl_safety</text>
  <text x="420" y="265" font-family="sans-serif" font-size="12" fill="#374151">DFA avoids accepting unsafe state</text>
</svg>

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
- `pctl` reports safety no longer satisfied.
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
- `pctl` currently behaves as a safety-so-far tracker over unsafe events.
- `reach_avoid` separately tracks whether the target was reached and whether the avoid label was ever seen.
- `ltl_safety` advances a DFA and reports violations when the automaton enters an accepting unsafe state.

The raw environment labels are the same. The constraint determines how those labels become safety state and metrics.
