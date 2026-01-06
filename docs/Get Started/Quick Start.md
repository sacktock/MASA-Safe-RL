Quick Start
===========

Installation
------------

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
conda env create --name masa --file conda-environment.yaml
conda activate masa
```
- Install dependencies:
```bash
pip install -e .
```

#### Installation with uv

Coming soon!

#### Installation with PyPI

Coming soon!

### Enabling GPU Acceleration with JAX (Optional)

MASA-Safe-RL relies on [JAX](https://docs.jax.dev/) for GPU acceleration. If you are only interested in the gymnasium wrappers and constraints API then you do not need to complete the following steps.

- **Linux x86_64/aarch64**: jax and jaxlib `0.4.30` should already be installed via the `requirements.txt`. You need to reinstall JAX based on your cuda driver compatibility. Do not use the ```-U``` option here!
```bash
pip install "jax[cuda13]"
```
or
```bash
pip install "jax[cuda12]"
```
- **Windows**: GPU acceletartion is also supported (experimentally) on Windows WSL x86_64. We strongly recommend using [Ubuntu 22.04](https://apps.microsoft.com/detail/9pn20msr04dw?hl=en-GB&gl=BE) or similar. You need to reinstall JAX based on your cuda driver compatibility. Do not use the ```-U``` option here!
```bash
pip install "jax[cuda13]"
```
or
```bash
pip install "jax[cuda12]"
```
- **MAC**: we recommend JAX with CPU. No further action is required if you correctly followed the earlier steps.

### Testing Your Installation

You can test your installation of masa via the command line,

```bash
python -m masa.run --env-id mini_pacman --algo q_learning --custom-cfgs mini_pacman --seed 0
```



