MASA Safe RL Docs
========================================

Welcome
-------


Welcome to MASA-Safe-RL, the Multi and Single Agent (MASA) Safe Reinforcement Learning library. The primary goal of this library is to develop a set of common constraints and environments for safe reinforcement learning research, built on top of the popular [gymnasium](https://gymnasium.farama.org/) interface. We span, CMPDs, probabilistic constraints, Reach-Avoid and LTL-Safety (DFA) properties.  

The library is in very early stage development and we greatly appreciate and encourage feedback from the community about what they would like to see implemented. Currently we provide a set of basic tabular algroithms for safe RL, but we provide a modular and resuable framework for developing more complex algorithms and constraints.


Organization
------------

The docs are currently organised as follows:

- **Get Started**: installation instructions and core API details (e.g., labelling function, cost function, make_env).
- **Constraints**: mathematical and usage details for the constraints currently provided in MASA.
- **Environments**: benchamrk environments currently provided in MASA.
- **Algorithms**: algorithm and logging API details and algorithms currently provided in MASA.
- **Other**: auxilliary wrappers, e.g., probabilistic shielding.

We recommend you follow the docs in the provided order starting from **Get Started** to learn about the core API and so on.

Citation
--------

If you use MASA-Safe-RL in your research please cite it in your publications.

```bibtex
@misc{Goodall2025MASASafeRL,
  title        = {{MASA-Safe-RL}: Multi and Single Agent Safe Reinforcement Learning},
  author       = {Goodall, Alexander W. and Adalat, Omar and Hamel De-le Court, Edwin and Belardinelli, Francesco},
  year         = {2025},
  howpublished = {\url{https://github.com/sacktock/MASA-Safe-RL/}},
  note         = {GitHub repository}
}
```

Developer Notes
---------------
- We need to add a short introduction to RL with the gymnasium interface and safe RL/RL with constraints in Core Concepts.
- We will discuss the labelling function and cost function API in core concepts.
- Tutorials will bed added on a rolling basis as the library is developed.

```{toctree}
:caption: Get Started
:hidden:

Get Started/Quick Start
Get Started/Core Concepts
Get Started/Basic Usage
```

```{toctree}
:caption: API
:hidden:

API/Constraints
API/Wrappers
API/Metrics
API/DFA
API/PCTL
```

```{toctree}
:caption: Environments
:hidden:

Environments/Multi Agent
Environments/Single Agent
```

```{toctree}
:caption: Algorithms
:hidden:

Algorithms/PPO Lagrangian
Algorithms/CPO
```

```{toctree}
:caption: Tutorials
:hidden:

Tutorials/Basics
Tutorials/Constraints
Tutorials/LTL-Safety
Tutorials/Wrappers
Tutorials/Environments
Tutorials/Algorithms
```

```{toctree}
:caption: Misc

:hidden:

Misc/Probabilistic Shielding
```