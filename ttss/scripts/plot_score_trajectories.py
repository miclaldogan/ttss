"""TTSS: plot threat score trajectories for sample videos — paper Figure 3.

Usage::

    python -m ttss.scripts.plot_score_trajectories --dummy
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ttss.visualization.score_plotter import ThreatScorePlotter


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--output-dir", default="evaluation/figures")
    return p


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(7)
    plotter = ThreatScorePlotter(dpi=300)

    samples = [
        ("Robbery", 180, 260),
        ("Shooting", 120, 200),
        ("Assault", 220, 290),
    ]

    for name, onset, end in samples:
        T = 350
        scores = np.clip(
            rng.random(T) * 0.3 + np.where(np.arange(T) >= onset - 30, np.linspace(0, 0.7, T), 0),
            0, 1,
        )
        save_path = Path(args.output_dir) / f"trajectory_{name.lower()}.png"
        plotter.plot_timeline(
            scores, crime_start=onset, crime_end=end,
            pre_crime_start=onset - 60, fps=30.0, frame_stride=8,
            title=f"Threat Score — {name}", save_path=save_path,
        )
        print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
