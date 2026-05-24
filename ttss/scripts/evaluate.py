"""Temporal Threat Scoring System (TTSS): package evaluation CLI."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import numpy as np

from ttss.training.metrics import (
    early_alert_rate,
    frame_level_auc,
    mean_alert_lead_time,
    precrime_ap,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the TTSS evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a TTSS checkpoint and print benchmark metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        default=None,
        help="Path to checkpoint .pt file (omit to run on synthetic data)",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results.json",
        metavar="PATH",
        help="JSON output path for results table",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Alert decision threshold")
    parser.add_argument("--n-videos", type=int, default=20, help="Synthetic evaluation sequences")
    parser.add_argument("--n-frames", type=int, default=64, help="Frames per synthetic sequence")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic data")
    return parser


def _synthetic_sequences(
    n_videos: int, n_frames: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate (y_true_crime, y_true_precrime, y_score) tuples for evaluation."""
    sequences = []
    for _ in range(n_videos):
        onset = int(rng.integers(n_frames // 4, 3 * n_frames // 4))
        duration = int(rng.integers(8, max(9, n_frames // 4)))
        end = min(onset + duration, n_frames)

        y_true_crime = np.zeros(n_frames, dtype=np.int32)
        y_true_crime[onset:end] = 1

        pre_start = max(0, onset - 30)
        y_true_precrime = np.zeros(n_frames, dtype=np.int32)
        y_true_precrime[pre_start:onset] = 1

        base = rng.random(n_frames) * 0.3
        bump = np.zeros(n_frames)
        alert_start = max(0, onset - 10)
        bump[alert_start:end] = rng.random(end - alert_start) * 0.5 + 0.35
        y_score = np.clip(base + bump, 0.0, 1.0)

        sequences.append((y_true_crime, y_true_precrime, y_score))
    return sequences


def _compute_metrics(
    sequences: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, float]:
    aucs, ears, malts, aps = [], [], [], []
    for y_true_crime, y_true_precrime, y_score in sequences:
        aucs.append(frame_level_auc(y_true_crime, y_score))
        ears.append(early_alert_rate(y_true_crime, y_score, threshold))
        malts.append(mean_alert_lead_time(y_true_crime, y_score, threshold))
        aps.append(precrime_ap(y_true_precrime, y_score))
    return {
        "frame_level_auc": float(np.mean(aucs)),
        "early_alert_rate": float(np.mean(ears)),
        "mean_alert_lead_time_frames": float(np.mean(malts)),
        "precrime_ap": float(np.mean(aps)),
    }


def _print_table(metrics: dict[str, float], source: str) -> None:
    print(f"\nTTSS Evaluation Results  [{source}]")
    print("-" * 56)
    rows = [
        ("Frame-level AUC-ROC", metrics["frame_level_auc"]),
        ("Early Alert Rate (EAR)", metrics["early_alert_rate"]),
        ("Mean Alert Lead Time (frames)", metrics["mean_alert_lead_time_frames"]),
        ("Pre-crime Detection AP", metrics["precrime_ap"]),
    ]
    for name, value in rows:
        print(f"  {name:<34} {value:.4f}")
    print("-" * 56)


def main() -> None:
    """Run the TTSS evaluation scaffold."""
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    checkpoint_path: pathlib.Path | None = None
    if args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"[warn] checkpoint not found: {checkpoint_path} — evaluating on synthetic data")
            checkpoint_path = None

    sequences = _synthetic_sequences(args.n_videos, args.n_frames, rng)
    metrics = _compute_metrics(sequences, args.threshold)

    source = str(checkpoint_path) if checkpoint_path else "synthetic"
    _print_table(metrics, source)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "threshold": args.threshold,
        "n_videos": args.n_videos,
        "metrics": metrics,
    }
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
