"""Temporal Threat Scoring System (TTSS): interactive demo CLI."""

from __future__ import annotations

import argparse

from ttss.inference.predictor import InferenceRequest, ThreatPredictor


def build_parser() -> argparse.ArgumentParser:
    """Build the TTSS demo CLI parser."""
    parser = argparse.ArgumentParser(description="Run a TTSS demo prediction.")
    parser.add_argument("--video-path", required=True, help="Input video path")
    return parser


def main() -> None:
    """Run the TTSS demo scaffold."""
    args = build_parser().parse_args()
    predictor = ThreatPredictor()
    result = predictor.predict_video(InferenceRequest(video_path=args.video_path))
    print(f"threat_score={result.threat_score:.4f}")
    print(f"temporal_label={result.temporal_label}")


if __name__ == "__main__":
    main()
