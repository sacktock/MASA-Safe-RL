# PPO

Source: `masa/algorithms/ppo/ppo.py`

`PPO` is MASA's main general-purpose deep RL algorithm. It shares the same actor-critic backbone as `A2C`, but optimizes the clipped PPO surrogate objective over multiple minibatch epochs for each rollout.

## Key Details

- collects on-policy rollouts with the shared `OnPolicyAlgorithm` machinery
- uses generalized advantage estimation
- optimizes a clipped surrogate objective controlled by `clip_range`
- trains over minibatches for multiple epochs per rollout

## Implementation Notes

The implementation uses the shared policy family in `masa/common/policies.py`, so the main differences from `A2C` are in the loss and optimization schedule rather than the model structure.

The code supports several Gymnasium action types through the shared policy and action-formatting code, including discrete and continuous control settings.

## When To Use It

Use `PPO` when:

- you want the main neural RL baseline in MASA
- you need a robust on-policy actor-critic method
- you want the algorithm used by the shielding examples before moving to shield-parameterized variants
