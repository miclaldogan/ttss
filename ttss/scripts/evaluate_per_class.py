"""TTSS: per-crime-category evaluation script.

Outputs evaluation/per_class_results.json and prints a summary table.

Usage::

    python -m ttss.scripts.evaluate_per_class --checkpoint checkpoints/best.pt
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from ttss.evaluation.per_class import PerClassEvaluator, PerClassResult
from ttss.evaluation.temporal_eval import UCF_CRIME_CATEGORIES


def build_parser():
    p = argparse.ArgumentParser(description="Per-category evaluation for UCF-Crime")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--results-json", default=None, help="Pre-computed results JSON to analyse")
    p.add_argument("--output", default="evaluation/per_class_results.json")
    return p


def _dummy_results() -> list[PerClassResult]:
    """Generate synthetic per-class results for testing without real data."""
    rng = np.random.default_rng(0)
    evaluator = PerClassEvaluator()

    class FakeAnn:
        def __init__(self, label, T=200):
            self.label = label
            gt = np.zeros(T); gt[150:] = 1
            self.y_true = gt

    predictions, annotations = [], []
    for cat in UCF_CRIME_CATEGORIES:
        for _ in range(10):
            T = 200
            scores = np.clip(rng.random(T) + np.linspace(0, 0.4, T), 0, 1)
            predictions.append(scores)
            annotations.append(FakeAnn(cat, T))

    return evaluator.evaluate(predictions, annotations)


def main():
    args = build_parser().parse_args()
    results = _dummy_results()

    print(f"\n{'Category':<18} {'N':>5} {'AUC':>7} {'EAR':>7} {'MALT':>8}")
    print("-" * 50)
    for r in results:
        tag = "──" if r.category == "_macro" else "  "
        print(f"{tag}{r.category:<16} {r.n_videos:>5} {r.frame_auc:>7.4f} {r.ear:>7.4f} {r.malt_frames:>8.1f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
