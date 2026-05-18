from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from masa.common.utils import make_env
from masa.envs.tabular.colour_grid_world import cost_fn, label_fn


DEFAULT_VIDEO_FOLDER = "videos/record_video_colour_grid_world"
ACTION_SEQUENCE = (1, 1, 2, 2, 2, 2, 4, 4)
TriggerMode = Literal["episode", "step"]


class RecordNextEpisodeEveryNSteps:
    """Episode trigger that records after crossing total-step intervals."""

    def __init__(self, interval: int):
        if interval < 1:
            raise ValueError("interval must be at least 1")
        self.interval = interval
        self.total_steps = 0
        self.next_threshold = interval
        self.pending_recordings = 0

    def observe_step(self) -> None:
        self.total_steps += 1
        while self.total_steps >= self.next_threshold:
            self.pending_recordings += 1
            self.next_threshold += self.interval

    def __call__(self, episode_id: int) -> bool:
        del episode_id
        if self.pending_recordings < 1:
            return False
        self.pending_recordings -= 1
        return True


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
        default=4,
        help="Number of short episodes to record.",
    )
    parser.add_argument(
        "--trigger-mode",
        choices=("episode", "step"),
        default="episode",
        help=(
            "Use 'episode' to record every Nth episode, or 'step' to record "
            "the next episode after every N total environment steps."
        ),
    )
    parser.add_argument(
        "--trigger-value",
        type=int,
        default=1,
        help=(
            "Interval for the selected trigger mode. With --trigger-mode episode, "
            "10 records episodes 10, 20, 30, ... . With --trigger-mode step, "
            "100 records the next episode after total steps 100, 200, 300, ... ."
        ),
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
    episodes: int = 4,
    trigger_mode: TriggerMode = "episode",
    trigger_value: int = 1,
    seed: int = 0,
    render_window_size: int = 256,
) -> list[Path]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if trigger_value < 1:
        raise ValueError("trigger_value must be at least 1")
    if render_window_size < 1:
        raise ValueError("render_window_size must be at least 1")

    step_trigger = None
    if trigger_mode == "episode":
        episode_trigger = lambda episode_id: (episode_id + 1) % trigger_value == 0
    elif trigger_mode == "step":
        step_trigger = RecordNextEpisodeEveryNSteps(trigger_value)
        episode_trigger = step_trigger
    else:
        raise ValueError("trigger_mode must be 'episode' or 'step'")

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
        record_video_episode_trigger=episode_trigger,
        video_folder=str(output_dir),
    )

    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            for action in ACTION_SEQUENCE:
                _, _, terminated, truncated, _ = env.step(action)
                if step_trigger is not None:
                    step_trigger.observe_step()
                if terminated or truncated:
                    break
    finally:
        env.close()

    recorded_files = sorted(output_dir.glob("*.mp4"))
    if not recorded_files:
        raise RuntimeError(
            f"No MP4 files were created in {output_dir}. "
            "Increase --episodes or lower --trigger-value for sparse triggers."
        )
    return recorded_files


def main() -> None:
    args = parse_args()
    recorded_files = run_recording(
        video_folder=args.video_folder,
        episodes=args.episodes,
        trigger_mode=args.trigger_mode,
        trigger_value=args.trigger_value,
        seed=args.seed,
        render_window_size=args.render_window_size,
    )

    print(f"Trigger: {args.trigger_mode} every {args.trigger_value}")
    print("Recorded video files:")
    for path in recorded_files:
        print(path)


if __name__ == "__main__":
    main()
