# Congestion

`CongestionMatrix` is a repeated routing game implemented in `masa.envs.multiagent.matrix.congestion`.

## Overview

- Class: `CongestionMatrix`
- Metadata name: `congestion_matrix_v0`
- Agents: `player_0` to `player_{N-1}` with `N=6` by default
- Actions: `RoadA=0`, `RoadB=1`
- Default parameters:
  `base_A=1`, `base_B=1`, `slope_A=1`, `slope_B=1`, `jam_threshold=5`, `jam_penalty=5`

Each agent picks one of two roads. Rewards are the negative congestion costs of the chosen road, with an extra penalty when a road load reaches the jam threshold.

## Safety Semantics

The environment labels jammed rounds with `jam` and `unsafe`. The default `cost_fn` returns binary cost `1` whenever a jam occurs.

## Observation Channels

The binary observation layout contains:

- one pair of channels per agent for the last chosen road,
- a one-hot encoding of the previous Road A load,
- a one-hot encoding of the previous Road B load,
- a final `jam` flag.

Its `label_fn` also emits derived labels such as `loadA_k`, `loadB_k`, `roadA_empty`, `roadB_empty`, and `balanced_load`.
