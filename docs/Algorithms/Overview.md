# Algorithms Overview

This section documents the algorithm classes currently present in the MASA codebase. The pages here are intentionally lightweight for now and focus on the core implementation ideas verified against the code.

## Implemented Algorithms

MASA currently contains three main groups of learning algorithms:

- tabular algorithms for discrete state and action spaces, including safety-aware variants
- neural on-policy actor-critic algorithms
- shield-aware PPO variants used with probabilistic shielding wrappers

The algorithms currently registered in the main plugin registry are:

| Algorithm | Family | Core idea | Safety mechanism |
| --- | --- | --- | --- |
| `QL` | Tabular | Standard one-step Q-learning baseline | None built into the update |
| `QL_Lambda` | Tabular | Q-learning with cost-penalized reward | Linear cost penalty |
| `SEM` | Tabular | Learns task and auxiliary safety-related tables | Safety-weighted action selection |
| `LCRL` | Tabular | Q-learning with absorbing-style violation value | Fixed violation return via `r_min` |
| `RECREG` | Tabular | Learns task and backup policies with overrides | Risk threshold and backup-action override |
| `PPO` | On-policy | Clipped actor-critic policy optimization | None built into the base algorithm |
| `A2C` | On-policy | Advantage actor-critic | None built into the base algorithm |

These are registered in `masa/plugins/supported.py`.

The multi-agent coordinators are available through the Python API rather than the current single-agent CLI registry:

| Algorithm | Family | Core idea | Safety mechanism |
| --- | --- | --- | --- |
| `IQL` | Multi-agent tabular | One existing `QL` learner per PettingZoo agent | None built into the update |
| `IQLLambda` | Multi-agent tabular | Independent Q-learning with selected CMG budget penalties | Fixed budget multipliers |
| `IQLLagrangian` | Multi-agent tabular | Independent penalized Q-learning with episode-level dual ascent | Adaptive multiplier per selected budget |

## Sections

- [Tabular](Tabular)
    - [Multi-Agent Tabular](Tabular/Multi%20Agent/)
- [On Policy](On%20Policy)