# Run from any working directory with make -C docs <target>.
UV ?= uv
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
RUN = cd "$(ROOT)" && $(UV) run --locked --only-group docs

.PHONY: help html build serve check clean
help:
	@echo "Targets: html (strict build), serve (preview), check (tests and links), clean"

html build:
	$(RUN) zensical build --strict --clean

serve:
	$(RUN) zensical serve

check:
	$(RUN) python -m unittest discover -s tests/docs -v
	$(RUN) python scripts/check_docs.py
	$(RUN) zensical build --strict --clean
	$(RUN) python scripts/check_docs.py --site site

clean:
	$(RUN) python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('site', '.cache', 'docs/_build')]"
