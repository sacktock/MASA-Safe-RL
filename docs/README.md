# MASA Docs

This folder contains the docmentation for [MASA-Safe-RL](https://github.com/sacktock/MASA-Safe-RL/)

## Building

The docs can be built with:
```sh
# 1. Without uv:
make html

#2 . If using Astral's uv package manager:
uv run make html
```

The documentation will be present at `docs/_build/html`, where you can open `index.html`.

Live reloading is supported with either:
```sh
#1. Without uv:
sphinx-autobuild docs docs/_build/html

#2. With uv:
uv run sphinx-autobuild docs docs/_build/html
```

Where the documentation should be auto-served at: `http://127.0.0.1:8000`