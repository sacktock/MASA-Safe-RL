MASA Safe RL Documentation documentation
========================================

Welcome to the Multi and Single Agent (MASA) Safe Reinforcement Learning library. The goal of this library to develop a set of common environments and interfaces for safe reinforcement learning research, spanning constrained MDPs, Probabilistic Constraints, Reach-Avoid, and LTL-safety (DFA) properties.

The documentation site is currently organised as follows:
- **Environments**: we list environments, alongside a description of our presupplied defined constraints, and benchmark results across all applicable algorithms to compare performance. 
- **Constraints**: we provide details of safety constraints, what assumptions they require (e.g. a labelling function, cost function).
- **Algorithms**: we provide details of algorithms that can be used, and what specific types of constraints they help satisfy.

Developer Notes:
- we could separate out **Benchmarks** into its own section
- we should add somewhere we discuss how Labelling/Cost functions can be defined (according to the library's way of defining these), perhaps in the **Constraints** section
- we could add a **Tutorials** section with walked examples of applying a labelling/cost function on an environment, applying the relevant constraint wrappers, using some of the algorithms and possibly automated hyperparameter tuning to get full results

```{toctree}
:hidden:
:caption: Environments

Environments/Multi Agent
Environments/Single Agent
```

```{toctree}
:caption: Constraints

Constraints/Multi Agent
Constraints/Constrained Markov Decision Process (CMDP)
Constraints/Reach Avoid
Constraints/LTL Safety
Constraints/PCTL
Constraints/Stepwise Probabilistic
```

```{toctree}
:caption: Algorithms

Algorithms/PPO Lagrangian
Algorithms/CPO
```

```{Benchmarks}
:caption: Benchmarks

Benchmarks/Benchmarks
```