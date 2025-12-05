# Multi and Single Agent Safe Reinforcement Learning

Welcome to the Multi and Single Agent (MASA) Safe Reinforcement Learning library. The goal of this library to develop a set of common environments and interfaces for safe reinforcement learning research, spanning constrained MDPs, Probabilistic Constraints, Reach-Avoid, and LTL-safety (DFA) properties. 

## Installation 
- Please install conda: [anaconda](https://anaconda.org/channels/anaconda/packages/conda/overview)
- Create a conda virtual environment:
```
conda create --name <env> --file requirements.txt
```

## How to run MASA
- You can run masa with the prebuilt run script. The script is not fully configurable so often it is better to create your own examples.
```
python -m masa.run --algo-configs bridge_crossing --algo ppo --seed 
```
- You can run examples from the ```\examples``` folder via:
```
python -m masa.examples.prob_shield_example
```
