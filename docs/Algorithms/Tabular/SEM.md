# SEM

Source: `masa/algorithms/tabular/sem.py`

`SEM` is a tabular algorithm that learns more than a single task-value table. In MASA it maintains three tables:

- `Q` for task return
- `D` as an auxiliary table updated using positive violation penalties
- `C` as an auxiliary table updated using negative violation penalties

## Key Details

- has separate learning rates and discount factors for the task and auxiliary tables
- updates the task table and the safety-related tables in parallel
- changes action selection using a safety weighting derived from `D` and `C`

## Action Selection Intuition

Unlike the simpler penalty methods, `SEM` does not rely only on changing the reward target. Instead, it adjusts action preferences directly using learned auxiliary tables. In the implementation this happens through a multiplicative weighting term built from clipped versions of `D` and `C`.

## When To Use It

Use `SEM` when:

- you want a more structured safe tabular method than plain penalization
- you want action selection to depend on explicit learned safety signals
- you want to separate task learning from auxiliary safety-related estimates
