# Multi-Agent Constraints

Here, we consider constraints and model interfaces applicable to multi-agent
environments.


## Labelled parallel environments

`LabelledParallelEnv` attaches proposition labels to each agent's `info` mapping
without changing observations, rewards, or actions.

::: masa.common.multi_agent.LabelledParallelEnv
    options:
      members: true

## Coalitions

A `Coalition` is an order-independent set of focal agents. Algorithms obtain the
canonical joint-action order from the wrapped environment's `possible_agents`.

::: masa.common.multi_agent.Coalition
    options:
      members: true
      skip_local_inventory: true

## Finite parallel environments

`TabularParallelEnv` is the multi-agent counterpart of the single-agent
`TabularEnv`. It adds an enumerable finite state and joint-transition-support
contract to PettingZoo's Parallel API. Labels remain the responsibility of
`LabelledParallelEnv`.

::: masa.envs.multiagent.TabularParallelEnv
    options:
      members: true
      skip_local_inventory: true

The coalition shielding wrapper expects this stack:

```text
TabularParallelEnv -> LabelledParallelEnv -> CoalitionLTLShield
```

See [Multi-agent coalition safety-game shielding](../../Shielding/Deterministic/Multi-Agent%20Coalition%20Shielding.md)
for synthesis and execution semantics.
