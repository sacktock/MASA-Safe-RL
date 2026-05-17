"""CLI entry point.

Example (generic researcher, ships with the ``performance`` spec):

    python -m masa.plotting \\
        --config masa/plotting/configs/example.yaml \\
        --stages all

Example (load extra specs from another module before resolving):

    python -m masa.plotting \\
        --config masa/plotting/examples/probshield/paper.yaml \\
        --specs-from masa.plotting.examples.probshield \\
        --stages process,render
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .io_utils import setup_logging, get_logger
from .pipeline import VALID_STAGES, Pipeline
from .specs import all_specs


def _parse_stages(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return tuple(VALID_STAGES)
    stages = tuple(s.strip() for s in raw.split(",") if s.strip())
    unknown = [s for s in stages if s not in VALID_STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; valid={list(VALID_STAGES)}")
    return stages


def _preload_spec_modules(modules: Sequence[str]) -> None:
    """Import each module so its register() calls run before specs are resolved."""
    log = get_logger()
    for name in modules:
        try:
            importlib.import_module(name)
            log.info(f"Loaded extra specs from {name}")
        except Exception as e:
            log.error(f"Could not import spec module {name!r}: {e}")
            raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m masa.plotting",
        description="Download W&B runs, aggregate to quantiles, render figures from PlotSpecs.",
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Path to a YAML config (see configs/example.yaml).")
    p.add_argument("--stages", default="all",
                   help="Comma separated subset of {download,process,render}, or 'all'.")
    p.add_argument("--specs", default=None,
                   help="Comma separated PlotSpec ids to render. Default: all registered.")
    p.add_argument("--specs-from", action="append", default=[],
                   metavar="MODULE",
                   help="Dotted Python module to import for additional spec registrations. "
                        "May be passed multiple times.")
    p.add_argument("--force-download", action="store_true",
                   help="Refetch every run even if a cached CSV exists.")
    p.add_argument("--force-process", action="store_true",
                   help="Rebuild the long form and quantile frames even if cached.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved plan and exit without W&B or disk writes.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    _preload_spec_modules(getattr(args, "specs_from", []))

    cfg = load_config(args.config)
    if args.force_download:
        cfg = replace(cfg, force_download=True)
    if args.force_process:
        cfg = replace(cfg, force_process=True)

    spec_ids = [s.strip() for s in args.specs.split(",") if s.strip()] if args.specs else None
    specs = all_specs(spec_ids)

    stages = _parse_stages(args.stages)

    pipeline = Pipeline(cfg)
    pipeline.run(specs=specs, stages=stages, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
