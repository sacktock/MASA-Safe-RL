# Shielded Algorithms

MASA provides two shielding approaches with different safety semantics and policy
interfaces.

```{toctree}
:maxdepth: 1

Shielded/Winning-Region Shielding
Shielded/Multi-Agent Coalition Shielding
Shielded/Probabilistic Shielding
Shielded/Parameterized PPO
Shielded/Parameterized PPO V2
```

## Winning-region safety-game shielding

[Winning-region safety-game shielding](Shielded/Winning-Region%20Shielding)
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

[Multi-agent coalition shielding](Shielded/Multi-Agent%20Coalition%20Shielding)
extends the same support-based construction to PettingZoo Parallel environments.
It supports centralised coalition joint actions and decentralised, independently
executable local masks while universally quantifying the actions of agents outside
the coalition.

## Probabilistic shielding

[Probabilistic Shielding](Shielded/Probabilistic%20Shielding) uses safety budgets
and projection onto safe action distributions.

It is based on **Probabilistic Shielding for Safe Reinforcement Learning** by
Edwin Hamel-De le Court, Francesco Belardinelli, and Alexander W. Goodall
([paper](https://arxiv.org/abs/2503.07671)).

The implementation classes live under:

```text
masa/prob_shield/
```

The parameterized PPO variants below are designed for the augmented action
interfaces used with probabilistic shielding. They are not required merely to use
a winning-region shield.
