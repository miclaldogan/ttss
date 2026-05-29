"""TTSS: score calibration script.

Plots a reliability diagram, reports ECE/MCE, and applies Platt scaling if
ECE > 0.05.

Usage::

    python -m ttss.scripts.calibrate_scores --dummy
    python -m ttss.scripts.calibrate_scores --checkpoint checkpoints/best.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")

from ttss.evaluation.calibration import reliability_diagram
from ttss.evaluation.statistics import PlattCalibrator


def build_parser():
    p = argparse.ArgumentParser(description="Score calibration for TTSS")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--output-dir", default="evaluation/figures")
    p.add_argument("--ece-threshold", type=float, default=0.05,
                   help="Apply Platt scaling when ECE exceeds this value")
    return p


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(42)
    T = 1000

    if args.dummy or args.checkpoint is None:
        y_true = (rng.random(T) > 0.6).astype(int)
        # Slightly overconfident model
        y_score = np.clip(rng.beta(3, 2, T) * y_true + rng.beta(2, 3, T) * (1 - y_true), 0, 1)
        print("Running in dummy mode with synthetic scores")
    else:
        raise NotImplementedError("Real checkpoint evaluation requires the full dataset pipeline")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Before calibration
    before = reliability_diagram(
        y_true, y_score, n_bins=10,
        save_path=out_dir / "calibration_before.png",
        title="Before Calibration",
    )
    print(f"Before calibration:  {before}")

    if before.ece > args.ece_threshold:
        print(f"ECE={before.ece:.4f} > threshold={args.ece_threshold:.2f} — applying Platt scaling")
        cal = PlattCalibrator()
        y_score_cal = cal.fit_transform(y_score, y_true)

        after = reliability_diagram(
            y_true, y_score_cal, n_bins=10,
            save_path=out_dir / "calibration_after.png",
            title="After Platt Scaling",
        )
        print(f"After  calibration:  {after}")
        print(f"ECE improvement: {before.ece:.4f} → {after.ece:.4f}  (Δ={after.ece - before.ece:+.4f})")
    else:
        print(f"ECE={before.ece:.4f} ≤ threshold — no calibration needed")

    print(f"Figures saved → {out_dir}/")


if __name__ == "__main__":
    main()
