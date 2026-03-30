# Dynamic Public Goods Game

`DPGGMatrix` is a repeated 2-player dynamic public-goods game implemented in `masa.envs.multiagent.matrix.dpgg`.

## Overview

- Class: `DPGGMatrix`
- Metadata name: `dpgg_matrix_v0`
- Agents: `player_0`, `player_1`
- Actions: `Contribute=0`, `Withhold=1`
- Default parameters:
  `contribution_cost=10`, `alpha=1.5`, `payout_rate=0.3`, `decay=0.0`, `initial_pot=0`, `pot_cap=500`, `pot_step=1`

The environment maintains a discretized public pot. Each round pays out a fraction of the current pot and then updates the pot according to contributions, amplification, and decay.

## Safety Semantics

The environment marks the joint action `Withhold, Withhold` as unsafe. Its `label_fn` emits `both_withhold` and `unsafe`, and the default `cost_fn` returns binary cost `1` on that outcome.

## Observation Channels

Observations are binary channels encoding:

- `player_0_contribute`
- `player_0_withhold`
- `player_1_contribute`
- `player_1_withhold`
- `pot_bit_0`, `pot_bit_1`, ... for the discretized pot index in LSB-first order

The `label_fn` also emits pot-state labels such as `pot_empty`, `pot_positive`, and `pot_nonempty`.
