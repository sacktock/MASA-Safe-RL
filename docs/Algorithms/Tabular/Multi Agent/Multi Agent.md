# Multi-agent Tabular Algorithms

This section documents MASA's first multi-agent learning coordinators. They are implemented under `masa/algorithms/tabular/multi_agent/` and operate on PettingZoo `ParallelEnv` instances that are wrapped with a MASA constrained Markov game (CMG).

The current algorithms use **independent learning**: every agent owns a separate instance of MASA's existing tabular `QL` learner. A lightweight coordinator collects a simultaneous joint action, steps the shared environment once, and sends each agent its own transition. 

For the time being, there is no joint-action Q-table, parameter sharing, or centralized critic.
