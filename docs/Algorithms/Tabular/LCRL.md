# LCRL

Source: `masa/algorithms/tabular/lcrl.py`

`LCRL` extends `QL`, but handles violations more aggressively than `QL_Lambda`. Rather than subtracting a tunable linear penalty, it gives violating transitions a fixed absorbing-style return based on `r_min / (1 - gamma)`.

## Key Details

- keeps the same tabular structure and exploration options as `QL`
- maps violations to a fixed long-run value controlled by `r_min`
- stops future bootstrapping through violating transitions

## Update Intuition

The main idea is that unsafe behaviour should not just be slightly worse than safe behaviour. Instead, it is mapped to an explicit absorbing-style value. That makes `LCRL` less about soft penalties and more about assigning a fixed long-run outcome to violations.

## When To Use It

Use `LCRL` when:

- you want violations to be strongly dominated in value
- a worst-case interpretation is more appropriate than a soft penalty
- you want a tabular baseline that is stricter than `QL_Lambda`
