"""Temporal Threat Scoring System (TTSS): pre-crime analysis CLI.

Reads timeline JSONs (or runs on synthetic data) and prints the MALT / EAR /
T-AUC table.  Results are written to ``evaluation/precrime_results.json``.

Usage examples::

    # Synthetic data (no real videos needed)
    python -m ttss.scripts.evaluate_precrime

    # Specify output path and threshold
    python -m ttss.scripts.evaluate_precrime --output evaluation/precrime_results.json --threshold 0.5

    # Read real timeline JSON files produced by evaluate.py
    python -m ttss.scripts.evaluate_precrime --timelines timeline1.json timeline2.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import numpy as np

from ttss.evaluation.precrime_metrics import PreCrimeMetrics


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-crime lead-time analysis: MALT, EAR curve, T-AUC table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timelines",
        nargs="*",
        metavar="PATH",
        default=None,
        help="Timeline JSON files (one per video).  Omit to run on synthetic data.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/precrime_results.json",
        metavar="PATH",
        help="JSON output path for results",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Alert threshold for MALT")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate (for MALT seconds)")
    parser.add_argument("--n-videos", type=int, default=20, help="Synthetic evaluation videos")
    parser.add_argument("--n-frames", type=int, default=64, help="Frames per synthetic video")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic data")
    return parser


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _synthetic_data(
    n_videos: int,
    n_frames: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int | None]]:
    """Return (scores, crime_starts).  Half the videos are non-crime."""
    scores: list[np.ndarray] = []
    crime_starts: list[int | None] = []
    for i in range(n_videos):
        if i < n_videos // 2:
            # Crime video.
            onset = int(rng.integers(n_frames // 4, 3 * n_frames // 4))
            base = rng.random(n_frames) * 0.3
            bump = np.zeros(n_frames)
            alert_start = max(0, onset - 15)
            bump[alert_start : onset + 10] = rng.random(onset + 10 - alert_start) * 0.5 + 0.35
            s = np.clip(base + bump, 0.0, 1.0)
            scores.append(s)
            crime_starts.append(onset)
        else:
            # Non-crime video — low scores throughout.
            s = rng.random(n_frames) * 0.35
            scores.append(s)
            crime_starts.append(None)
    return scores, crime_starts


# ---------------------------------------------------------------------------
# Timeline JSON loading
# ---------------------------------------------------------------------------


def _load_timelines(
    paths: list[str],
) -> tuple[list[np.ndarray], list[int | None]]:
    """Load score arrays and crime_starts from timeline JSON files.

    Expected JSON schema (as written by ``evaluate.py``)::

        {
          "scores": [<float>, ...],
          "crime_start": <int> | null
        }
    """
    scores: list[np.ndarray] = []
    crime_starts: list[int | None] = []
    for p in paths:
        data = json.loads(pathlib.Path(p).read_text())
        scores.append(np.asarray(data["scores"], dtype=float))
        crime_starts.append(data.get("crime_start"))
    return scores, crime_starts


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_malt(malt_frames: float, malt_seconds: float) -> None:
    print(f"\nMALT: {malt_frames:.1f} frames  ({malt_seconds:.3f} s)")


def _print_ear_curve(thresholds: np.ndarray, ear_values: np.ndarray) -> None:
    print("\nEAR Curve:")
    print(f"  {'Threshold':>10}  {'EAR':>8}")
    print("  " + "-" * 22)
    for t, e in zip(thresholds, ear_values):
        print(f"  {t:>10.2f}  {e:>8.4f}")


def _print_temporal_roc(temporal_roc: dict[int, object]) -> None:
    print("\nTemporal ROC (T-AUC@L):")
    print(f"  {'L (frames)':>12}  {'T-AUC':>8}")
    print("  " + "-" * 24)
    for L in sorted(temporal_roc):
        print(f"  {L:>12d}  {temporal_roc[L].auc:>8.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    metrics = PreCrimeMetrics(default_fps=args.fps)

    # Load or generate data.
    if args.timelines:
        scores, crime_starts = _load_timelines(args.timelines)
    else:
        scores, crime_starts = _synthetic_data(args.n_videos, args.n_frames, rng)

    # Crime-only lists for MALT and EAR.
    crime_scores = [s for s, c in zip(scores, crime_starts) if c is not None]
    crime_onsets = [c for c in crime_starts if c is not None]

    # Compute metrics.
    malt_result = metrics.mean_alert_lead_time(
        crime_scores, crime_onsets, fps=args.fps, threshold=args.threshold
    )
    ear_result = metrics.ear_curve(crime_scores, crime_onsets)
    troc = metrics.temporal_roc(scores, crime_starts)

    # Print.
    _print_malt(malt_result.malt_frames, malt_result.malt_seconds)
    _print_ear_curve(ear_result.thresholds, ear_result.ear_values)
    _print_temporal_roc(troc)

    # Write JSON.
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_videos": len(scores),
        "fps": args.fps,
        "threshold": args.threshold,
        "malt": {
            "malt_frames": malt_result.malt_frames,
            "malt_seconds": malt_result.malt_seconds,
            "per_video_frames": malt_result.per_video_frames,
        },
        "ear_curve": {
            "thresholds": ear_result.thresholds.tolist(),
            "ear_values": ear_result.ear_values.tolist(),
        },
        "temporal_roc": {
            str(L): {
                "fpr": troc[L].fpr.tolist(),
                "tpr": troc[L].tpr.tolist(),
                "auc": troc[L].auc,
            }
            for L in sorted(troc)
        },
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
