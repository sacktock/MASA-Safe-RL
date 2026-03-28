# MASA Docs

This folder contains the docmentation for [MASA-Safe-RL](https://github.com/sacktock/MASA-Safe-RL/)

The documentation is managed by the repository root `pyproject.toml` and `uv.lock`. Run the following commands from the repository root; there is no separate docs-local `uv` environment to sync.

## Building with uv

```sh
uv sync --group docs
uv run --locked sphinx-build -b html docs docs/_build/html
```

The documentation will be present at `docs/_build/html`, where you can open `index.html`.

## Live reload with uv

```sh
uv run --locked sphinx-autobuild docs docs/_build/html
```

The docs will be served at `http://127.0.0.1:8000`.

## Building without uv

The docs can still be built from the repository root with:

```sh
make -C docs html
```

Live reloading without `uv` is:

```sh
sphinx-autobuild docs docs/_build/html
```
