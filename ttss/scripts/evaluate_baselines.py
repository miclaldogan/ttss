"""Temporal Threat Scoring System (TTSS): baseline evaluation CLI.

Usage
-----
    python -m ttss.scripts.evaluate_baselines --baseline all --split test
    python -m ttss.scripts.evaluate_baselines --baseline sultani2018 rtfm

Output
------
Prints a per-baseline results table to stdout and writes
``evaluation/baseline_results.json`` with one record per (baseline, video):

    {
      "baseline":        str,
      "video_id":        str,
      "frame_auc":       float,
      "early_alarm_rate": float
    }
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import numpy as np

from ttss.baselines import REGISTRY
from ttss.training.metrics import early_alert_rate, frame_level_auc

_AVAILABLE_BASELINES = list(REGISTRY)


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate TTSS baselines on UCF-Crime (or synthetic) data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=["all"],
        metavar="NAME",
        help=f"Baselines to evaluate.  'all' expands to {_AVAILABLE_BASELINES}.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split (used as a label; evaluates on synthetic data when "
        "UCF-Crime is not found).",
    )
    parser.add_argument(
        "--output",
        default="evaluation/baseline_results.json",
        metavar="PATH",
        help="JSON output path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Alert decision threshold.",
    )
    parser.add_argument(
        "--n-videos",
        type=int,
        default=10,
        help="Synthetic videos per baseline.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=64,
        help="Frames per synthetic video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for synthetic ground-truth labels.",
    )
    return parser


def _resolve_baselines(names: list[str]) -> list[str]:
    """Expand 'all' and validate requested baseline names."""
    if "all" in names:
        return list(_AVAILABLE_BASELINES)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown baseline(s): {unknown}.  Available: {_AVAILABLE_BASELINES}"
        )
    return names


def _synthetic_labels(
    n_videos: int, n_frames: int, rng: np.random.Generator
) -> list[tuple[str, np.ndarray]]:
    """Generate ``(video_id, y_true_crime)`` pairs for synthetic evaluation."""
    pairs = []
    for i in range(n_videos):
        video_id = f"synthetic://video_{i:03d}"
        onset = int(rng.integers(n_frames // 4, 3 * n_frames // 4))
        duration = int(rng.integers(8, max(9, n_frames // 4)))
        end = min(onset + duration, n_frames)
        y_true = np.zeros(n_frames, dtype=np.int32)
        y_true[onset:end] = 1
        pairs.append((video_id, y_true))
    return pairs


def _evaluate_baseline(
    baseline_name: str,
    videos: list[tuple[str, np.ndarray]],
    threshold: float,
    n_frames: int,
) -> list[dict]:
    """Run one baseline over all videos and return per-video result records."""
    cls = REGISTRY[baseline_name]
    predictor = cls(n_frames=n_frames)

    records = []
    for video_id, y_true_crime in videos:
        y_score = predictor.predict_video(video_id)
        # Pad or trim to match y_true length
        if len(y_score) != len(y_true_crime):
            y_score = np.resize(y_score, len(y_true_crime))

        records.append(
            {
                "baseline": baseline_name,
                "video_id": video_id,
                "frame_auc": round(float(frame_level_auc(y_true_crime, y_score)), 4),
                "early_alarm_rate": round(
                    float(early_alert_rate(y_true_crime, y_score, threshold)), 4
                ),
            }
        )
    return records


def _print_summary(all_records: list[dict], split: str) -> None:
    """Print a per-baseline summary table."""
    by_baseline: dict[str, list[dict]] = {}
    for r in all_records:
        by_baseline.setdefault(r["baseline"], []).append(r)

    print(f"\nBaseline Evaluation Results  [split={split}]")
    print("-" * 56)
    header = f"  {'Baseline':<24} {'AUC-ROC':>8}  {'EAR':>8}"
    print(header)
    print("-" * 56)
    for bname, records in sorted(by_baseline.items()):
        mean_auc = float(np.mean([r["frame_auc"] for r in records]))
        mean_ear = float(np.mean([r["early_alarm_rate"] for r in records]))
        print(f"  {bname:<24} {mean_auc:>8.4f}  {mean_ear:>8.4f}")
    print("-" * 56)


def main() -> None:
    """Run baseline evaluation and write JSON results."""
    args = build_parser().parse_args()

    baseline_names = _resolve_baselines(args.baseline)
    rng = np.random.default_rng(args.seed)
    videos = _synthetic_labels(args.n_videos, args.n_frames, rng)

    all_records: list[dict] = []
    for name in baseline_names:
        records = _evaluate_baseline(name, videos, args.threshold, args.n_frames)
        all_records.extend(records)

    _print_summary(all_records, args.split)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "threshold": args.threshold,
        "results": all_records,
    }
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
