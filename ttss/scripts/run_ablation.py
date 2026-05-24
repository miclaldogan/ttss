"""Temporal Threat Scoring System (TTSS): ablation study runner.

Usage
-----
    python -m ttss.scripts.run_ablation --experiment arch
    python -m ttss.scripts.run_ablation --experiment window
    python -m ttss.scripts.run_ablation --experiment fusion
    python -m ttss.scripts.run_ablation --experiment arch window fusion

Each experiment sweeps its predefined variants, runs a synthetic forward
pass to verify all models instantiate and produce valid ThreatPrediction
outputs, computes frame-level AUC-ROC and EAR on synthetic labels, and
writes ``evaluation/ablation_{experiment}.json``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import numpy as np
import torch

from ttss.training.ablation import (
    ARCH_VARIANTS,
    FUSION_SWEEP_CONFIGS,
    WINDOW_SWEEP_CONFIGS,
    AblationConfig,
    build_ablation_model,
)
from ttss.training.metrics import early_alert_rate, frame_level_auc

_EXPERIMENTS = ("arch", "window", "fusion")

# Synthetic evaluation constants
_N_VIDEOS = 8
_N_FRAMES = 32
_BATCH_SIZE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the ablation runner CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run TTSS ablation experiments and write evaluation JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        nargs="+",
        default=["arch"],
        choices=[*_EXPERIMENTS, "all"],
        metavar="NAME",
        help=f"Experiment(s) to run: {_EXPERIMENTS} or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation",
        metavar="DIR",
        help="Directory to write ablation JSON files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Alert decision threshold for EAR.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for synthetic data.",
    )
    return parser


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _synthetic_labels(
    n_videos: int, n_frames: int, rng: np.random.Generator
) -> list[tuple[str, np.ndarray]]:
    """Generate ``(video_id, y_true_crime)`` pairs."""
    pairs = []
    for i in range(n_videos):
        vid = f"synthetic://video_{i:03d}"
        onset = int(rng.integers(n_frames // 4, 3 * n_frames // 4))
        end = min(onset + int(rng.integers(4, max(5, n_frames // 8))), n_frames)
        y_true = np.zeros(n_frames, dtype=np.int32)
        y_true[onset:end] = 1
        pairs.append((vid, y_true))
    return pairs


def _run_model_on_video(
    model: torch.nn.Module, input_dim: int, n_frames: int, seed: int
) -> np.ndarray:
    """Run a synthetic forward pass and return per-frame scores as numpy."""
    model.eval()
    with torch.no_grad():
        x = torch.rand(_BATCH_SIZE, n_frames, input_dim, generator=torch.Generator().manual_seed(seed))
        pred = model(x)
        # pred is ThreatPrediction; frame_scores has shape (B, T)
        scores = pred.frame_scores[0].cpu().numpy().astype(np.float32)
    return scores


# ---------------------------------------------------------------------------
# Per-experiment runners
# ---------------------------------------------------------------------------


def _run_arch(
    threshold: float,
    rng: np.random.Generator,
    seed: int,
) -> list[dict]:
    """Sweep all six architecture variants."""
    labels = _synthetic_labels(_N_VIDEOS, _N_FRAMES, rng)
    records = []
    for name, cfg in ARCH_VARIANTS.items():
        model = build_ablation_model(cfg)
        aucs, ears = [], []
        for i, (vid, y_true) in enumerate(labels):
            y_score = _run_model_on_video(model, cfg.input_dim, _N_FRAMES, seed + i)
            if len(y_score) != len(y_true):
                y_score = np.resize(y_score, len(y_true))
            aucs.append(frame_level_auc(y_true, y_score))
            ears.append(early_alert_rate(y_true, y_score, threshold))
        records.append(
            {
                "variant": name,
                "description": cfg.description,
                "input_dim": cfg.input_dim,
                "bidirectional": cfg.bidirectional,
                "use_attention": cfg.use_attention,
                "precrime_weight": cfg.precrime_weight,
                "consistency_lambda": cfg.consistency_lambda,
                "frame_auc": round(float(np.mean(aucs)), 4),
                "early_alarm_rate": round(float(np.mean(ears)), 4),
            }
        )
    return records


def _run_window(
    threshold: float,
    rng: np.random.Generator,
    seed: int,
) -> list[dict]:
    """Sweep pre-crime window K ∈ {0, 30, 60, 90, 120, 150}."""
    labels = _synthetic_labels(_N_VIDEOS, _N_FRAMES, rng)
    records = []
    for cfg in WINDOW_SWEEP_CONFIGS:
        model = build_ablation_model(cfg)
        aucs, ears = [], []
        for i, (vid, y_true) in enumerate(labels):
            y_score = _run_model_on_video(model, cfg.input_dim, _N_FRAMES, seed + i)
            if len(y_score) != len(y_true):
                y_score = np.resize(y_score, len(y_true))
            aucs.append(frame_level_auc(y_true, y_score))
            # EAR: restrict to pre-crime window of size K
            onset_idx = int(np.where(y_true == 1)[0][0]) if np.any(y_true) else _N_FRAMES
            k = cfg.precrime_window_k
            pre_start = max(0, onset_idx - k)
            if onset_idx > pre_start:
                window_true = np.zeros(_N_FRAMES, dtype=np.int32)
                window_true[pre_start:onset_idx] = 1
                # Re-use onset from trimmed window
                ears.append(early_alert_rate(window_true, y_score, threshold))
            else:
                ears.append(0.0)
        records.append(
            {
                "K": cfg.precrime_window_k,
                "description": cfg.description,
                "frame_auc": round(float(np.mean(aucs)), 4),
                "early_alarm_rate": round(float(np.mean(ears)), 4),
            }
        )
    return records


def _run_fusion(
    threshold: float,
    rng: np.random.Generator,
    seed: int,
) -> list[dict]:
    """Sweep feature fusion strategies (concat / additive / attention)."""
    labels = _synthetic_labels(_N_VIDEOS, _N_FRAMES, rng)
    records = []
    for cfg in FUSION_SWEEP_CONFIGS:
        model = build_ablation_model(cfg)
        aucs, ears = [], []
        for i, (vid, y_true) in enumerate(labels):
            y_score = _run_model_on_video(model, cfg.input_dim, _N_FRAMES, seed + i)
            if len(y_score) != len(y_true):
                y_score = np.resize(y_score, len(y_true))
            aucs.append(frame_level_auc(y_true, y_score))
            ears.append(early_alert_rate(y_true, y_score, threshold))
        records.append(
            {
                "fusion": cfg.fusion,
                "input_dim": cfg.input_dim,
                "description": cfg.description,
                "frame_auc": round(float(np.mean(aucs)), 4),
                "early_alarm_rate": round(float(np.mean(ears)), 4),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_arch_table(records: list[dict]) -> None:
    print("\nAblation: Architecture Variants")
    print("-" * 64)
    print(f"  {'Variant':<28} {'AUC-ROC':>8}  {'EAR':>8}")
    print("-" * 64)
    for r in records:
        print(f"  {r['variant']:<28} {r['frame_auc']:>8.4f}  {r['early_alarm_rate']:>8.4f}")
    print("-" * 64)


def _print_window_table(records: list[dict]) -> None:
    print("\nAblation: Pre-crime Window K Sweep")
    print("-" * 48)
    print(f"  {'K':>6}  {'AUC-ROC':>8}  {'EAR':>8}")
    print("-" * 48)
    for r in records:
        print(f"  {r['K']:>6}  {r['frame_auc']:>8.4f}  {r['early_alarm_rate']:>8.4f}")
    print("-" * 48)


def _print_fusion_table(records: list[dict]) -> None:
    print("\nAblation: Feature Fusion Strategy")
    print("-" * 52)
    print(f"  {'Strategy':<16}  {'Input dim':>9}  {'AUC-ROC':>8}  {'EAR':>8}")
    print("-" * 52)
    for r in records:
        print(
            f"  {r['fusion']:<16}  {r['input_dim']:>9}  "
            f"{r['frame_auc']:>8.4f}  {r['early_alarm_rate']:>8.4f}"
        )
    print("-" * 52)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run requested ablation experiments and write JSON results."""
    args = build_parser().parse_args()
    experiments: list[str] = args.experiment
    if "all" in experiments:
        experiments = list(_EXPERIMENTS)

    rng = np.random.default_rng(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for exp in experiments:
        if exp == "arch":
            records = _run_arch(args.threshold, rng, args.seed)
            _print_arch_table(records)
        elif exp == "window":
            records = _run_window(args.threshold, rng, args.seed)
            _print_window_table(records)
        else:  # fusion
            records = _run_fusion(args.threshold, rng, args.seed)
            _print_fusion_table(records)

        out_path = output_dir / f"ablation_{exp}.json"
        payload = {
            "experiment": exp,
            "timestamp": ts,
            "threshold": args.threshold,
            "results": records,
        }
        with out_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
