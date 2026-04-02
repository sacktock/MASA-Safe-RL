# RECREG

Source: `masa/algorithms/tabular/recreg.py`

`RECREG` is the most intervention-oriented tabular algorithm currently in MASA. It combines task learning with an explicit backup policy and uses a safety estimate to decide when to override the task action.

## Core Components

The implementation maintains:

- a task Q-table `Q`
- a backup policy table `B`
- a safety estimate used to test the task action against a finite-horizon risk threshold

At action time, the algorithm first proposes an action from the task policy. If that action exceeds the current finite-horizon risk threshold, it is replaced by an action from the backup policy.

## Supported Modes

`RECREG` supports three modes for estimating safety:

- `exact`: uses the environment's exact transition model
- `model_based`: estimates transitions online from counts and performs model checking on the learned model
- `model_free`: learns a finite-horizon unsafe-probability table directly

In model-based mode, the implementation supports exact or statistical model checking.

## Safety-Relevant Behaviour

- overridden risky actions are pushed toward a pessimistic target
- in `model_based` and `model_free`, the backup policy is updated online using cost-aware targets
- in `exact`, the backup policy is initialized from value iteration on the exact model
- override rates are logged during rollout and evaluation
- DFA-based counterfactual transitions are supported for LTL-safety product environments

## When To Use It

Use `RECREG` when:

- you want explicit intervention rather than only cost shaping
- you want a learned backup policy available at decision time
- you want finite-horizon probabilistic safety checks to gate actions online
