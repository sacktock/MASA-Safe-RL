# Chicken

`ChickenMatrix` is a repeated 2-player Chicken game implemented in `masa.envs.multiagent.matrix.chicken`.

## Overview

- Class: `ChickenMatrix`
- Metadata name: `chicken_matrix_v0`
- Agents: `player_0`, `player_1`
- Actions: `Swerve=0`, `Straight=1`
- Default payoffs: `T=3`, `R=2`, `S=1`, `P=0`

The stage game is:

- `Swerve, Swerve -> (R, R)`
- `Swerve, Straight -> (S, T)`
- `Straight, Swerve -> (T, S)`
- `Straight, Straight -> (P, P)`

## Safety Semantics

The environment treats `Straight, Straight` as a crash. Its `label_fn` emits `crash` and `unsafe`, and the default `cost_fn` assigns binary cost `1` to that unsafe outcome.

## Observation Channels

Observations are binary channels representing the previous joint action and whether the last round ended in a crash:

- `player_0_swerve`
- `player_0_straight`
- `player_1_swerve`
- `player_1_straight`
- `crash`
