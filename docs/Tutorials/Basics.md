# Basics

These tutorials are the shortest path from a fresh checkout to a working MASA experiment.

- [First MASA Experiment](Basics/First%20MASA%20Experiment) shows how to preview an environment, build a MASA wrapper stack with `make_env`, inspect labels and constraint metrics, and run a tiny Q-learning smoke experiment.
- [Labels, Costs, and Infos](Basics/Labels%20Costs%20and%20Infos) slows down the environment loop and inspects observations, rewards, labels, costs, constraint metrics, termination, and truncation.
- [Wrapper Stack](Basics/Wrapper%20Stack) builds the same constrained environment with `make_env` and by manually applying each MASA wrapper.
- [Constraints Tour](Basics/Constraints%20Tour) compares the registered single-agent constraints on the same environment and action scripts.

Runnable notebooks:

- [notebooks/tutorials/01_first_masa_experiment.ipynb](../../notebooks/tutorials/01_first_masa_experiment.ipynb)
- [notebooks/tutorials/02_labels_costs_and_infos.ipynb](../../notebooks/tutorials/02_labels_costs_and_infos.ipynb)
- [notebooks/tutorials/03_wrapper_stack.ipynb](../../notebooks/tutorials/03_wrapper_stack.ipynb)
- [notebooks/tutorials/04_constraints_tour.ipynb](../../notebooks/tutorials/04_constraints_tour.ipynb)

```{toctree}
:maxdepth: 1
:hidden:

Basics/First MASA Experiment
Basics/Labels Costs and Infos
Basics/Wrapper Stack
Basics/Constraints Tour
```
