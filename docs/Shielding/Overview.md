# Shielded Algorithms

MASA provides two shielding approaches with different safety semantics and policy
interfaces.


## Winning-region safety-game shielding

[Winning-region safety-game shielding](Deterministic/Winning-Region%20Shielding.md)
computes the product of a finite transition model and a bad-prefix safety DFA.
It permits only actions whose entire modeled successor support remains inside the
winning region.

The implementation lives under:

```text
masa/deterministic_shield/
```

For single-agent Gymnasium environments, two policy interfaces are available:

- `PreemptiveLTLShield` exposes the safe-action mask before action selection.
- `PostposedLTLShield` preserves safe proposals and replaces unsafe proposals.

Postposed replacement can use the default lowest-index safe action,
`random_safe(...)`, `highest_score(...)`, or a custom callback. Random replacement
does **not** introduce a safety-risk budget: it samples only from the already
computed safe set.

[Multi-agent coalition shielding](Deterministic/Multi-Agent%20Coalition%20Shielding.md)
extends the same support-based construction to PettingZoo Parallel environments.
It supports centralised coalition joint actions and decentralised, independently
executable local masks while universally quantifying the actions of agents outside
the coalition.

## Probabilistic shielding

[Probabilistic Shielding](Probabilistic/Probabilistic%20Shielding.md) uses safety budgets
and projection onto safe action distributions.

It is based on **Probabilistic Shielding for Safe Reinforcement Learning** by
Edwin Hamel-De le Court, Francesco Belardinelli, and Alexander W. Goodall
([paper](https://arxiv.org/abs/2503.07671)).

The implementation classes live under:

```text
masa/prob_shield/
```

The parameterized PPO variants used are designed for the augmented action
interfaces used with probabilistic shielding.

### Supporting Infrastructure

Several components are not standalone learning algorithms, but they are important for understanding how MASA algorithms work:

- `masa/common/on_policy_algorithm.py`: shared rollout, return, and GAE logic for `A2C` and `PPO`
- `masa/common/policies.py`: actor-critic networks and action distributions
- `masa/prob_shield/eventual_discounted_vi.py`: value iteration used by shielding utilities and by `RECREG` in exact mode
- `masa/prob_shield/interval_bound_vi.py`: interval-bound value iteration for safety analysis
