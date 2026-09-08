# Working on the MASA documentation

The site is built with **Zensical**, configured in the repository-root
`zensical.toml`. Dependencies live in the root `pyproject.toml` docs group and
`uv.lock`; there is no separate docs environment or requirements file.

## Preview and build

Run these commands from the repository root:

```sh
uv sync --locked --only-group docs
uv run --locked --only-group docs zensical serve
```

The preview is served at `http://127.0.0.1:8000`. For a production build and checks:

```sh
uv run --locked --only-group docs python -m unittest discover -s tests/docs -v
uv run --locked --only-group docs python scripts/check_docs.py
uv run --locked --only-group docs zensical build --strict --clean
uv run --locked --only-group docs python scripts/check_docs.py --site site
```

The output is `site/`. `make -C docs html`, `make -C docs serve`, and
`docs\make.bat html` / `docs\make.bat serve` are convenience wrappers around uv.
`make -C docs clean` removes generated output, not source files.

Only documentation dependencies are installed. mkdocstrings reads Python source
statically, so building the site does not import MASA or require TensorFlow, JAX,
or a GPU runtime. Changing `masa/` also triggers the preview's file watcher.

## Authoring

Write ordinary Markdown and add every published page to `project.nav` in
`zensical.toml`. Use relative links ending in `.md`; percent-encode spaces and
parentheses. Existing `.html` page paths are retained through
`use_directory_urls = false`. Sphinx-generated member anchors can differ after
migration; link to qualified API identifiers instead of hard-coding new anchors.

API references use mkdocstrings:

```markdown
::: masa.common.constraints.base.Constraint
    options:
      members: true
```

Google-style docstrings are supported. `scripts/docs_docstrings.py` adapts legacy
reStructuredText roles and math in rendered docstrings only; new docstrings can
use Markdown directly. For a second rendering of an already documented object,
set `skip_local_inventory: true` to preserve the canonical API link target.

Use `$...$` / `$$...$$` for equations, `!!! note` for admonitions, standard pipe
tables, and Markdown images or `<figure markdown="1">` for captioned figures.
No MyST fences, `eval-rst`, Sphinx toctrees, or Sphinx installation are needed.

## Theme and assets

The original MASA logos live in `docs/assets/images/`. Keep copies aligned with
`images/` when updating the brand. Existing tutorial figures remain beside their
pages. `docs/_static/custom.css` supplies the navy/teal/green palettes, responsive
landing-page cards, accessible focus states, and bordered API signatures. Edit
those variables and semantic classes rather than copying theme templates.

Light, dark, and system modes are available. Fonts are local system fonts.
Equations use pinned MathJax 3.2.2 from jsDelivr; `_static/mathjax.js` re-typesets
content after instant navigation. Math rendering therefore requires network
access to that CDN, while the documentation text remains readable offline.

## Continuous integration and deployment

The docs workflow runs the same tests, strict build, and local URL/anchor checks
for branches and pull requests. It uploads the built site as a preview artifact.
Only a push to `main` (or a manual run on `main`) deploys to GitHub Pages. A
migration branch cannot replace the live site. Site, repository, and edit URLs
are explicit in `zensical.toml`; adjust them when publishing from another fork.
