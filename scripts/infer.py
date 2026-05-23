"""Minimal inference entrypoint for TTSS experiments."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TTSS inference on a video segment."
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--crime-start",
        type=int,
        required=True,
        help="Crime start frame",
    )
    parser.add_argument(
        "--crime-end",
        type=int,
        required=True,
        help="Crime end frame",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        required=True,
        help="Frame to inspect",
    )
    return parser


def main() -> None:
    from ttss import assign_temporal_label

    args = build_parser().parse_args()
    label = assign_temporal_label(
        frame_index=args.frame_index,
        crime_start_frame=args.crime_start,
        crime_end_frame=args.crime_end,
    )
    print("TTSS inference scaffold")
    print(f"video={args.video}")
    print(f"frame_index={args.frame_index}")
    print(f"temporal_label={label}")


if __name__ == "__main__":
    main()
