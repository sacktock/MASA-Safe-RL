"""CLI entrypoint: ``python masa/plotting/run.py``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python masa/plotting/run.py",
        description="Plot MASA training metrics from TensorBoard or W&B.",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--tensorboard", dest="tb_logdir", type=str,
                     help="Path to TensorBoard log directory")
    src.add_argument("--wandb", dest="wandb_project", type=str,
                     help="W&B project name")

    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (default: your default entity)")
    parser.add_argument("--run-filter", type=str, nargs="+", default=None,
                        help="Substring filter(s) on run / directory name (space-separated)")

    parser.add_argument("-m", "--metrics", nargs="+", default=None,
                        help="Metric keys to plot (default: all available)")
    parser.add_argument("--list-metrics", action="store_true",
                        help="List available metrics and exit")

    parser.add_argument("-w", "--smooth-weight", type=float, default=0.6,
                        help="EMA smoothing weight [0, 1) (default: 0.6)")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save path for the figure (e.g. results.pdf)")
    parser.add_argument("--group-seeds", action="store_true",
                        help="Aggregate seeds into q25/q50/q75 bands (TensorBoard only)")
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--no-axes", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args(argv)

    from masa.plotting.sources import TensorBoardSource, WandBSource

    if args.tb_logdir:
        source = TensorBoardSource(args.tb_logdir)
    else:
        source = WandBSource(args.wandb_project, entity=args.entity)

    if args.list_metrics:
        available = source.list_metrics()
        if not available:
            print("No metrics found.")
            sys.exit(1)
        print("Available metrics:")
        for m in available:
            print(f"  {m}")
        sys.exit(0)

    from masa.plotting.api import plot_run

    kwargs: dict = dict(
        metrics=args.metrics,
        smooth_weight=args.smooth_weight,
        title=args.title,
        output_path=args.output,
        show_legend=not args.no_legend,
        show_axes=not args.no_axes,
        dpi=args.dpi,
        run_filter=args.run_filter,
        group_seeds=args.group_seeds,
    )

    if args.tb_logdir:
        kwargs["tensorboard_logdir"] = args.tb_logdir
    else:
        kwargs["wandb_project"] = args.wandb_project
        kwargs["wandb_entity"] = args.entity

    fig = plot_run(**kwargs)

    if args.output:
        print(f"Saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
