# MASA-Safe-RL

Welcome to MASA-Safe-RL, the Multi and Single Agent (MASA) Safe Reinforcement Learning library. The primary goal of this library is to develop a set of common constraints and environments for safe reinforcement learning research, built on top of the popular [gymnasium](https://gymnasium.farama.org/) interface. We span, CMPDs, probabilistic constraints, Reach-Avoid and LTL-Safety (DFA) properties.  

The library is in very early stage development and we greatly appreciate and encourage feedback from the community about what they would like to see implemented. Currently we provide a set of basic tabular algroithms for safe RL, but we provide a modular and resuable framework for developing more complex algorithms and constraints.

If you use MASA-Safe-RL in your research please cite it in your publications.

```bibtex
@misc{Goodall2025MASASafeRL,
  title        = {{MASA-Safe-RL}: Multi and Single Agent Safe Reinforcement Learning},
  author       = {Goodall, Alexander W. and Adalat, Omar and Hamel De-le Court, Edwin and Belardinelli, Francesco},
  year         = {2025},
  howpublished = {\url{https://github.com/sacktock/MASA-Safe-RL/}},
  note         = {GitHub repository}
}
```

## Quick Start

### Installation

#### Prequisites

Python 3.8+ is required but we recommend Python 3.10 (later Python versions may not be supported).

#### Installation with conda
- Install conda, e.g., via [anaconda](https://anaconda.org/channels/anaconda/packages/conda/overview).
- Clone the repo:
```bash
git clone https://github.com/sacktock/MASA-Safe-RL.git
cd MASA-Safe-RL
```
- Create a conda virtual environment:
```bash
conda create --name masa --file conda-environment.yml
conda activate masa
```
- Install dependencies:
```bash
pip install -r requirements.txt
```

#### Installation with uv

Coming soon!

#### Installation with PyPI

Coming soon!

### Enabling GPU Acceleration with JAX

MASA-Safe-RL relies on [JAX](https://docs.jax.dev/) for GPU acceleration. If you are only interested in the gymnasium wrappers and constraints API then you do not need to complete the following steps.

- **Linux x86_64/aarch64**: jax and jaxlib `0.4.30` should already be installed via the `requirements.txt`. You need to reinstall JAX based on your cuda driver compatibility,
```bash
pip install -U "jax[cuda13]"
```
or
```bash
pip install -U "jax[cuda12]"
```
-**Windows**: GPU acceletartion is also supported (experimentally) on Windows WSL x86_64. We strongly recommend using [Ubuntu 22.04](https://apps.microsoft.com/detail/9pn20msr04dw?hl=en-GB&gl=BE) or similar. You need to reinstall JAX based on your cuda driver compatibility,
```bash
pip install -U "jax[cuda13]"
```
or
```bash
pip install -U "jax[cuda12]"
```
-**MAC**: we recommend JAX with CPU. No further action is required if you correctly followed the earlier steps.

## How to run MASA
- You can run masa with the prebuilt `run` script. The script is not fully configurable so it is often better to create your own examples.
```
python -m masa.run --env-id bridge_crossing --algo ppo --custom-cfgs bridge_crossing --seed 0
```
- You can run examples from the ```\examples``` folder via:
```
python -m masa.examples.prob_shield_example
```

## Getting in Touch

MASA-Safe-RL is primarliy managed by [Alex Goodall](https://github.com/sacktock) and [Omar Adalat](https://github.com/nightly). For correspondence in the early stages of the library we prefer you contact us directly via email (a.goodall22@imperial.ac.uk), rather than raising issues on GitHub directly.

## License

MASA-Safe-RL is released under Apache License 2.0.
