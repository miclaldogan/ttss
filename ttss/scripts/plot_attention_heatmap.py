"""TTSS: plot attention weight heatmap — paper Figure 4.

Usage::

    python -m ttss.scripts.plot_attention_heatmap --dummy
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ttss.visualization.attention_plotter import AttentionHeatmapPlotter


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--output-dir", default="evaluation/figures")
    return p


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(3)
    plotter = AttentionHeatmapPlotter(dpi=300)

    T = 200
    crime_start = 130
    raw_attn = np.abs(rng.normal(0, 1, T))
    raw_attn[crime_start - 20:crime_start + 10] *= 3.0
    attn = raw_attn / raw_attn.sum()
    scores = np.clip(rng.random(T) + np.where(np.arange(T) >= crime_start - 20, 0.4, 0), 0, 1)

    save_path = Path(args.output_dir) / "attention_heatmap.png"
    plotter.plot(attn, scores=scores, crime_start=crime_start,
                 fps=30.0, frame_stride=8, save_path=save_path)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
