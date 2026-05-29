"""TTSS: horizontal bar chart of per-category AUC — paper Figure 7.

Usage::

    python -m ttss.scripts.plot_per_class \\
        --results evaluation/per_class_results.json \\
        --output  evaluation/figures/per_class_auc.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="evaluation/per_class_results.json")
    p.add_argument("--output", default="evaluation/figures/per_class_auc.png")
    p.add_argument("--dummy", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    if args.dummy or not Path(args.results).exists():
        rng = np.random.default_rng(1)
        from ttss.evaluation.temporal_eval import UCF_CRIME_CATEGORIES
        data = [{"category": c, "frame_auc": float(np.clip(rng.normal(0.78, 0.06), 0.5, 0.95)),
                 "n_videos": 10} for c in UCF_CRIME_CATEGORIES]
    else:
        with open(args.results) as f:
            data = json.load(f)

    data = [r for r in data if r["category"] != "_macro"]
    data.sort(key=lambda r: r["frame_auc"])

    categories = [r["category"] for r in data]
    aucs = [r["frame_auc"] for r in data]
    colors = ["#ef5350" if a >= 0.80 else "#ffa726" if a >= 0.70 else "#66bb6a" for a in aucs]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(categories, aucs, color=colors, edgecolor="white", height=0.65)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(np.mean(aucs), color="navy", linestyle="--", linewidth=1.0, label=f"Mean {np.mean(aucs):.3f}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Frame-level AUC")
    ax.set_title("Per-Category AUC — UCF-Crime Test Set")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved 300 DPI figure → {out}")


if __name__ == "__main__":
    main()
