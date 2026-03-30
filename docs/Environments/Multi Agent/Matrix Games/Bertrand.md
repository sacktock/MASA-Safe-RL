# Bertrand

`BertrandMatrix` is a repeated 2-player price-competition game implemented in `masa.envs.multiagent.matrix.bertrand`.

## Overview

- Class: `BertrandMatrix`
- Metadata name: `bertrand_matrix_v0`
- Agents: `player_0`, `player_1`
- Actions: `High=0`, `Low=1`
- Default payoffs: `T=8`, `R=5`, `S=0`, `P=0`

The stage game is:

- `High, High -> (R, R)`
- `High, Low -> (S, T)`
- `Low, High -> (T, S)`
- `Low, Low -> (P, P)`

## Safety Semantics

The environment labels `Low, Low` as a price war. Its `label_fn` emits `price_war` and `unsafe`, and the default `cost_fn` returns a binary cost of `1` whenever `unsafe` is present.

## Observation Channels

Observations are binary channels representing the previous joint action and whether the last round ended in a price war:

- `player_0_high`
- `player_0_low`
- `player_1_high`
- `player_1_low`
- `price_war`
