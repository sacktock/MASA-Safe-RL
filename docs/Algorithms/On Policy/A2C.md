# A2C

Source: `masa/algorithms/a2c/a2c.py`

`A2C` is MASA's advantage actor-critic implementation. It is built on the shared `OnPolicyAlgorithm` scaffold and uses the same actor-critic policy family as PPO.

## Key Details

- collects on-policy rollouts using the shared rollout buffer
- computes returns and generalized advantage estimates
- performs one gradient update on the actor and critic per rollout batch
- can optionally normalize advantages

## Implementation Notes

The class delegates most rollout handling to `masa/common/on_policy_algorithm.py`. That shared scaffold handles vectorized environments, action formatting, bootstrapping on truncation, and return or advantage computation. `A2C` mainly defines the actor-critic loss and the one-update optimization step.

The default policy class is `PPOPolicy`, so `A2C` and `PPO` share the same network family even though they use different optimization objectives.

## When To Use It

Use `A2C` when:

- you want a simple on-policy actor-critic baseline
- you want a lighter update scheme than PPO
- you want to compare unclipped policy-gradient style training against PPO
