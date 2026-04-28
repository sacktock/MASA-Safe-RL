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

## Sections

- [Tabular](Tabular)
- [On Policy](On%20Policy)
- [Shielded](Shielded)

## Supporting Infrastructure

Several components are not standalone learning algorithms, but they are important for understanding how MASA algorithms work:

- `masa/common/on_policy_algorithm.py`: shared rollout, return, and GAE logic for `A2C` and `PPO`
- `masa/common/policies.py`: actor-critic networks and action distributions
- `masa/prob_shield/eventual_discounted_vi.py`: value iteration used by shielding utilities and by `RECREG` in exact mode
- `masa/prob_shield/interval_bound_vi.py`: interval-bound value iteration for safety analysis