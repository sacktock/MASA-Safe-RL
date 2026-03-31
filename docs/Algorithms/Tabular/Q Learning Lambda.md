# Q Learning Lambda

Source: `masa/algorithms/tabular/q_learning_lambda.py`

`QL_Lambda` extends `QL` with a linear cost penalty. Instead of learning only from task reward, it subtracts `cost_lambda * cost` from the reward target.

## Key Details

- keeps the same tabular structure and exploration options as `QL`
- introduces a cost-weighting parameter `cost_lambda`
- suppresses future bootstrapping after a violating transition

## Update Intuition

The algorithm is still Q-learning, but the immediate target becomes a penalized reward. In practice this means unsafe behaviour is discouraged by making it less valuable, rather than by explicitly shielding or overriding actions.

This makes `QL_Lambda` the most direct penalty-based safe tabular baseline in the codebase.

## When To Use It

Use `QL_Lambda` when:

- you want a simple reward-penalty approach
- expected cost penalties are an acceptable safety signal
- you want a minimal change from standard Q-learning
