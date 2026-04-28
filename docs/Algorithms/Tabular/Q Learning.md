# Q Learning

Source: `masa/algorithms/tabular/q_learning.py`

`QL` is the base tabular Q-learning implementation in MASA. It assumes discrete observation and action spaces and learns a single Q-table with one-step temporal-difference updates and a max backup.

## Key Details

- supports Boltzmann and epsilon-greedy exploration
- stores a small transition buffer and updates the Q-table from collected transitions
- uses the standard target `reward + gamma * max_a' Q(s', a')`, with no future bootstrap on terminal transitions

## Safety-Relevant Behaviour

If the environment uses a DFA-based cost function, `QL` generates counterfactual transitions for every automaton state in the product MDP. This is important for LTL-safety settings because it lets the learner update from all automaton-state interpretations of the observed transition, not only the one actually visited.

Outside that case, the algorithm still records step cost and violation information during rollout, but the optimization target itself remains standard task Q-learning.

## When To Use It

Use `QL` as the baseline tabular method when you want:

- a simple discrete-state baseline
- a reference point for comparing the safe tabular variants
- the shared rollout and exploration behaviour used by the other tabular algorithms
