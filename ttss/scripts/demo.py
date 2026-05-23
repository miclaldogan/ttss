"""Temporal Threat Scoring System (TTSS): interactive demo CLI."""

from __future__ import annotations

import argparse

from ttss.models.ttss_pipeline import TTPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the TTSS demo CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run a TTSS threat-scoring demo on a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, metavar="PATH", help="Input video path")
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Early-warning score threshold",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Sample every N-th frame",
    )
    return parser


def main() -> None:
    """Run the TTSS demo scaffold."""
    args = build_parser().parse_args()
    pipeline = TTPipeline(early_warning_threshold=args.threshold)
    timeline = pipeline.predict(args.video, frame_stride=args.stride)
    print(f"sequence_score={timeline.sequence_score:.4f}")
    print(f"frames_analysed={len(timeline.frame_ids)}")
    flagged = sum(timeline.early_warning_flags)
    print(f"early_warning_frames={flagged}")
    if flagged:
        print("EARLY WARNING TRIGGERED")


if __name__ == "__main__":
    main()
