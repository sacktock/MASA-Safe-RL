# Media Streaming

`media_streaming` is a small tabular Markov decision process modelling adaptive streaming with a finite playback buffer. It exposes a
full transition matrix via `get_transition_matrix()`.

## State and Actions

- Observation space: `Discrete(20)`
- Action space: `Discrete(2)`
- Start state: `10`

The state is the current buffer occupancy. State `0` represents an empty buffer and is labelled `"empty"`. The initial state is the
middle of the buffer.

The two actions correspond to different download rates:

- `0`: slow rate (`0.1`)
- `1`: fast rate (`0.9`)

The environment also uses a fixed outgoing playback rate of `0.7`.

## Transition Model

From a non-boundary state, the next buffer level can move:

- up by one,
- stay the same,
- down by one,

depending on whether incoming data arrives and whether playback consumes buffered content.

The transition probabilities are derived from:

- incoming rate determined by the selected action,
- outgoing rate `0.7`.

At the empty buffer boundary, the buffer cannot decrease further. At the full boundary, the implementation forces the incoming rate to
`1.0`, which keeps the top state well-defined and creates a deterministic safe end component for downstream safety methods.

## Reward, Labels, and Costs

The environment separates performance and safety in a very direct way:

- `label_fn(obs)` returns `{"start"}` at the initial state and `{"empty"}` at state `0`.
- `cost_fn(labels)` returns `1.0` exactly when `"empty"` is present.
- reward is `0.0` for action `0` and `-1.0` for action `1`.

So:

- empty-buffer events are the default safety violations,
- higher bitrate is expensive in the reward signal,
- the agent must trade off aggressive buffering against the risk of starvation.

## Episode Semantics

`media_streaming` never terminates or truncates internally. In practice, it is usually wrapped with `TimeLimit` or constructed via
`make_env(..., max_episode_steps=40)` as in the default MASA configuration.

This makes it a clean continuing-task benchmark for CMDP-style constraints and probabilistic safety objectives.
