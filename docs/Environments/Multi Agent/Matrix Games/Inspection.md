# Inspection

`InspectionMatrix` is a repeated 2-player inspection game implemented in `masa.envs.multiagent.matrix.inspection`.

## Overview

- Class: `InspectionMatrix`
- Metadata name: `inspection_matrix_v0`
- Agents: `player_0` (inspector), `player_1` (inspectee)
- Actions:
  `player_0`: `NotInspect=0`, `Inspect=1`
  `player_1`: `Comply=0`, `Violate=1`
- Default parameters: `b=5`, `f=10`, `c=2`, `h=4`, `v=3`

The stage game is:

- `NotInspect, Comply -> (0, 0)`
- `NotInspect, Violate -> (-h, b)`
- `Inspect, Comply -> (-c, 0)`
- `Inspect, Violate -> (v-c, -f)`

## Safety Semantics

The unsafe event is an undetected violation, corresponding to `NotInspect, Violate`. Its `label_fn` emits `undetected_violation` and `unsafe`, and the default `cost_fn` assigns binary cost `1` in that case.

## Observation Channels

Observations are binary channels representing the previous actions and whether the last round contained an undetected violation:

- `inspector_notinspect`
- `inspector_inspect`
- `inspectee_comply`
- `inspectee_violate`
- `undetected_violation`
