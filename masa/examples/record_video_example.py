from __future__ import annotations

import argparse
from pathlib import Path

from masa.common.utils import make_env
from masa.envs.tabular.colour_grid_world import cost_fn, label_fn


DEFAULT_VIDEO_FOLDER = "videos/record_video_colour_grid_world"
ACTION_SEQUENCE = (1, 1, 2, 2, 2, 2, 4, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record short ColourGridWorld rollouts with Gymnasium's RecordVideo wrapper."
    )
    parser.add_argument(
        "--video-folder",
        default=DEFAULT_VIDEO_FOLDER,
        help=f"Directory for generated MP4 files. Defaults to {DEFAULT_VIDEO_FOLDER!r}.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of short episodes to record.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base reset seed. Episode index is added to this value.",
    )
    parser.add_argument(
        "--render-window-size",
        type=int,
        default=256,
        help="Square RGB frame size requested from the environment renderer.",
    )
    return parser.parse_args()


def run_recording(
    *,
    video_folder: str | Path = DEFAULT_VIDEO_FOLDER,
    episodes: int = 2,
    seed: int = 0,
    render_window_size: int = 256,
) -> list[Path]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if render_window_size < 1:
        raise ValueError("render_window_size must be at least 1")

    output_dir = Path(video_folder)
    env = make_env(
        "colour_grid_world",
        "cmdp",
        len(ACTION_SEQUENCE),
        label_fn=label_fn,
        cost_fn=cost_fn,
        budget=0.0,
        env_kwargs={
            "render_mode": "rgb_array",
            "render_window_size": render_window_size,
        },
        record_video=True,
        record_video_episode_trigger=lambda episode_id: True,
        video_folder=str(output_dir),
    )

    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            for action in ACTION_SEQUENCE:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()

    recorded_files = sorted(output_dir.glob("*.mp4"))
    if not recorded_files:
        raise RuntimeError(f"No MP4 files were created in {output_dir}.")
    return recorded_files


def main() -> None:
    args = parse_args()
    recorded_files = run_recording(
        video_folder=args.video_folder,
        episodes=args.episodes,
        seed=args.seed,
        render_window_size=args.render_window_size,
    )

    print("Recorded video files:")
    for path in recorded_files:
        print(path)


if __name__ == "__main__":
    main()
