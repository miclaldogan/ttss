"""TTSS: comprehensive evaluation — ALL metrics from the best checkpoint.

Uses the official per-frame ground-truth labels (gt-ucf.npy, 1,114,144 frames)
instead of weak bag-level labels.  Produces a single results JSON with every
metric the paper needs, plus a figure bundle.

Metrics reported
----------------
Primary
  frame_auc          Frame-level AUC-ROC (main UCF-Crime benchmark)

Pre-crime
  ear                Early Alert Rate @ threshold 0.5
  malt_frames        Mean Alert Lead Time in frames
  malt_seconds       MALT in seconds (MALT / fps)
  precrime_ap        Average precision for pre-crime frame detection
  t_auc_L0/30/60/90  Temporal AUC conditioned on lead-time ≥ L

Per-category
  per_category       AUC/EAR/MALT for each of the 13 crime classes

Statistical
  auc_ci_95          Bootstrap 95% CI on frame AUC
  ece                Expected Calibration Error (Platt-calibrated output)

Usage::

    python -m ttss.scripts.evaluate_all_metrics \\
        --checkpoint  outputs/ttss/checkpoints/best.pt \\
        --features-dir data/features \\
        --gt-file      data/raw/UCF-Crime/annotations/gt-ucf.npy \\
        --test-list    data/splits/ucf_crime_test.txt \\
        --output       evaluation/full_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference on pre-extracted features
# ---------------------------------------------------------------------------


def _run_inference(
    model: torch.nn.Module,
    features_dir: Path,
    test_list: Path,
    device: str,
    clip_length: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Run BiLSTM over all test feature files.

    Returns
    -------
    (score_arrays, video_ids)
    Each score_array has shape (T_sampled,) — variable length per video.
    """
    video_ids = [l.strip() for l in test_list.read_text().splitlines() if l.strip()]
    score_arrays: list[np.ndarray] = []
    found_ids: list[str] = []

    model.eval()
    with torch.no_grad():
        for vid_id in video_ids:
            npz_path = features_dir / "test" / f"{vid_id}.npz"
            if not npz_path.exists():
                log.warning("Missing features: %s", npz_path)
                continue

            data = np.load(npz_path, allow_pickle=True)
            yolo = data["yolo_features"].astype(np.float32)   # (T, 8)
            vit  = data["vit_features"].astype(np.float32)    # (T, 768)
            feats = np.concatenate([yolo, vit], axis=1)       # (T, 776)

            # Run in chunks of clip_length to handle variable-length sequences
            T = len(feats)
            chunks = [feats[i:i + clip_length] for i in range(0, T, clip_length)]
            scores_list: list[float] = []
            for chunk in chunks:
                if len(chunk) < clip_length:
                    pad = np.zeros((clip_length - len(chunk), feats.shape[1]), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)
                x = torch.from_numpy(chunk).unsqueeze(0).to(device)
                result = model(x)
                preds = result.frame_scores if hasattr(result, "frame_scores") else result
                scores_list.extend(preds.squeeze(0).detach().cpu().tolist()[:len(feats[len(scores_list):])])

            score_arrays.append(np.array(scores_list[:T], dtype=np.float32))
            found_ids.append(vid_id)

    log.info("Ran inference on %d / %d test videos", len(found_ids), len(video_ids))
    return score_arrays, found_ids


# ---------------------------------------------------------------------------
# Map gt-ucf.npy slices to per-video arrays
# ---------------------------------------------------------------------------


def _split_gt_by_video(
    gt_flat: np.ndarray,
    score_arrays: list[np.ndarray],
    frame_stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split the flat gt array into per-video slices matching score_arrays.

    gt-ucf.npy has per-frame labels at original fps (30).
    score_arrays have labels sampled at frame_stride.
    For each video with T_sampled predicted frames, the corresponding
    gt slice covers T_sampled * frame_stride original frames.
    """
    gt_per_video: list[np.ndarray] = []
    scores_aligned: list[np.ndarray] = []
    offset = 0

    for scores in score_arrays:
        T_sampled = len(scores)
        T_original = T_sampled * frame_stride

        if offset + T_original > len(gt_flat):
            T_original = len(gt_flat) - offset
            T_sampled = T_original // frame_stride
            if T_sampled == 0:
                break

        gt_video = gt_flat[offset:offset + T_original:frame_stride][:T_sampled]
        gt_per_video.append(gt_video.astype(int))
        scores_aligned.append(scores[:T_sampled])
        offset += T_original

    return gt_per_video, scores_aligned


# ---------------------------------------------------------------------------
# Compute all metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(
    gt_per_video: list[np.ndarray],
    score_arrays: list[np.ndarray],
    video_ids: list[str],
    fps: float = 30.0,
    frame_stride: int = 8,
    threshold: float = 0.5,
) -> dict:
    from ttss.training.metrics import (
        frame_level_auc, early_alert_rate, mean_alert_lead_time, precrime_ap,
    )
    from ttss.evaluation.precrime_metrics import PreCrimeMetrics
    from ttss.evaluation.per_class import PerClassEvaluator, PerClassResult
    from ttss.evaluation.statistics import bootstrap_ci, bonferroni_correction
    from ttss.evaluation.calibration import reliability_diagram
    import re

    # Flatten all frames for global metrics
    y_true_all = np.concatenate(gt_per_video)
    y_score_all = np.concatenate(score_arrays)

    # ── Primary ──────────────────────────────────────────────────────────
    global_auc  = frame_level_auc(y_true_all, y_score_all)
    global_ear  = early_alert_rate(y_true_all, y_score_all, threshold)
    global_malt = mean_alert_lead_time(y_true_all, y_score_all, threshold)

    # pre-crime AP (all frames before first anomaly onset globally)
    onset_idx = np.where(y_true_all == 1)[0]
    global_precrime_ap = 0.0
    if len(onset_idx) > 0:
        onset = int(onset_idx[0])
        y_pre = np.zeros_like(y_true_all)
        y_pre[:onset] = 1
        global_precrime_ap = precrime_ap(y_pre, y_score_all)

    log.info("Frame AUC: %.4f  EAR: %.4f  MALT: %.1f frames (%.2f s)  pre-AP: %.4f",
             global_auc, global_ear, global_malt,
             global_malt / (fps / frame_stride), global_precrime_ap)

    # ── Bootstrap CI on AUC ──────────────────────────────────────────────
    log.info("Computing bootstrap CI (n=500)...")
    ci = bootstrap_ci(y_true_all, y_score_all, frame_level_auc, n_bootstrap=500)
    log.info("AUC 95%% CI: %.4f [%.4f, %.4f]", ci.mean, ci.lower, ci.upper)

    # ── Temporal ROC (T-AUC @ lead-times) ────────────────────────────────
    crime_starts = []
    for gt in gt_per_video:
        onset_i = np.where(gt == 1)[0]
        crime_starts.append(int(onset_i[0]) if len(onset_i) > 0 else None)

    pcm = PreCrimeMetrics(default_fps=fps / frame_stride)
    temporal_roc = pcm.temporal_roc(score_arrays, crime_starts, lead_times=[0, 30, 60, 90])
    ear_curve = pcm.ear_curve(score_arrays, [c or 0 for c in crime_starts])
    malt_result = pcm.mean_alert_lead_time(
        score_arrays, [c or 0 for c in crime_starts], fps=fps / frame_stride
    )

    t_auc = {str(L): round(r.auc, 4) for L, r in temporal_roc.items()}
    log.info("T-AUC @ L: %s", t_auc)

    # ── Per-category breakdown ────────────────────────────────────────────
    class _Ann:
        def __init__(self, label, y_true):
            self.label = label
            self.y_true = y_true

    anns = []
    for vid_id, gt in zip(video_ids, gt_per_video):
        match = re.match(r'^([A-Za-z_]+?)(\d)', vid_id.replace("_x264", ""))
        label = match.group(1).rstrip("_") if match else "Unknown"
        anns.append(_Ann(label, gt))

    evaluator = PerClassEvaluator(threshold=threshold, fps=fps / frame_stride)
    per_cat = evaluator.evaluate(score_arrays, anns)
    log.info("Per-category AUC:")
    for r in per_cat:
        log.info("  %-18s AUC=%.4f  EAR=%.4f  MALT=%.1f  n=%d",
                 r.category, r.frame_auc, r.ear, r.malt_frames, r.n_videos)

    # ── Calibration ───────────────────────────────────────────────────────
    cal_result = reliability_diagram(y_true_all, y_score_all, n_bins=10)
    log.info("Calibration — ECE=%.4f  MCE=%.4f", cal_result.ece, cal_result.mce)

    # ── Assemble results dict ─────────────────────────────────────────────
    return {
        "primary": {
            "frame_auc": round(global_auc, 4),
            "frame_auc_ci_lower": round(ci.lower, 4),
            "frame_auc_ci_upper": round(ci.upper, 4),
            "ear": round(global_ear, 4),
            "malt_frames": round(global_malt, 2),
            "malt_seconds": round(global_malt / (fps / frame_stride), 2),
            "precrime_ap": round(global_precrime_ap, 4),
        },
        "temporal_auc": t_auc,
        "ear_curve": {
            "thresholds": ear_curve.thresholds.tolist(),
            "ear_values": ear_curve.ear_values.tolist(),
        },
        "malt": {
            "malt_frames": round(malt_result.malt_frames, 2),
            "malt_seconds": round(malt_result.malt_seconds, 2),
        },
        "calibration": {
            "ece": round(cal_result.ece, 4),
            "mce": round(cal_result.mce, 4),
        },
        "per_category": [r.to_dict() for r in per_cat],
        "n_test_videos": len(gt_per_video),
        "n_test_frames": int(len(y_true_all)),
    }


# ---------------------------------------------------------------------------
# Generate figures
# ---------------------------------------------------------------------------


def _generate_figures(results: dict, score_arrays: list, gt_per_video: list,
                       video_ids: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg")

    from ttss.inference.visualizer import plot_roc_curve, plot_ear_curve
    from ttss.evaluation.precrime_metrics import PreCrimeMetrics
    from ttss.training.metrics import frame_level_auc

    y_true_all = np.concatenate(gt_per_video)
    y_score_all = np.concatenate(score_arrays)

    # ROC curve
    from ttss.evaluation.precrime_metrics import _roc_from_scores
    roc = _roc_from_scores(y_true_all.astype(int), y_score_all)
    plot_roc_curve(roc.fpr, roc.tpr, roc.auc,
                   save_path=out_dir / "roc_curve.png", title="TTSS ROC — UCF-Crime")

    # EAR curve
    ear = results["ear_curve"]
    plot_ear_curve(ear["thresholds"], ear["ear_values"],
                   save_path=out_dir / "ear_curve.png")

    # Per-category bar chart
    import matplotlib.pyplot as plt
    per_cat = [r for r in results["per_category"] if r["category"] != "_macro"]
    per_cat.sort(key=lambda r: r["frame_auc"])
    cats = [r["category"] for r in per_cat]
    aucs = [r["frame_auc"] for r in per_cat]
    colors = ["#ef5350" if a >= 0.80 else "#ffa726" if a >= 0.70 else "#66bb6a" for a in aucs]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(cats, aucs, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(np.mean(aucs), color="navy", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Frame-level AUC")
    ax.set_title(f"Per-Category AUC  (global={results['primary']['frame_auc']:.3f})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_category_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Score timeline for first anomaly video
    for vid_id, gt, scores in zip(video_ids, gt_per_video, score_arrays):
        if gt.max() > 0:
            from ttss.visualization.score_plotter import ThreatScorePlotter
            onset = int(np.where(gt == 1)[0][0])
            plotter = ThreatScorePlotter(dpi=300)
            plotter.plot_timeline(
                scores, frame_labels=gt, crime_start=onset,
                fps=30.0, frame_stride=8,
                title=f"Threat Score — {vid_id}",
                save_path=out_dir / f"timeline_{vid_id}.png",
            )
            break  # just one example

    log.info("Figures saved to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTSS full evaluation — all metrics")
    p.add_argument("--checkpoint",   default="outputs/ttss/checkpoints/best.pt")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--gt-file",      default="data/raw/UCF-Crime/annotations/gt-ucf.npy")
    p.add_argument("--test-list",    default="data/splits/ucf_crime_test.txt")
    p.add_argument("--output",       default="evaluation/full_results.json")
    p.add_argument("--figures-dir",  default="evaluation/figures")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame-stride", type=int, default=8)
    p.add_argument("--clip-length",  type=int, default=64)
    p.add_argument("--threshold",    type=float, default=0.5)
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Load model
    from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor
    model = BiLSTMThreatPredictor(input_dim=776).to(args.device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded checkpoint: %s (epoch=%s, val_auc=%.4f)",
                 ckpt_path, ckpt.get("epoch", "?"), ckpt.get("val_auc", 0.0))
    else:
        log.warning("Checkpoint not found — using random weights: %s", ckpt_path)

    # Run inference
    features_dir = Path(args.features_dir)
    test_list    = Path(args.test_list)
    if not features_dir.exists() or not test_list.exists():
        log.error("Features or test list not found. Run extract_features.py first.")
        return

    score_arrays, video_ids = _run_inference(
        model, features_dir, test_list, args.device, args.clip_length
    )
    if not score_arrays:
        log.error("No features found. Did extract_features.py complete?")
        return

    # Load and align ground-truth labels
    gt_flat = np.load(args.gt_file)
    log.info("gt-ucf.npy: %d frames total", len(gt_flat))
    gt_per_video, score_arrays = _split_gt_by_video(gt_flat, score_arrays, args.frame_stride)

    # Compute everything
    results = compute_all_metrics(
        gt_per_video, score_arrays, video_ids,
        fps=30.0, frame_stride=args.frame_stride, threshold=args.threshold,
    )

    # Save JSON
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("\nResults saved → %s", out)

    # Print summary table
    p = results["primary"]
    print(f"\n{'='*55}")
    print(f"  TTSS Evaluation — UCF-Crime Test Set ({results['n_test_videos']} videos)")
    print(f"{'='*55}")
    print(f"  Frame AUC:       {p['frame_auc']:.4f}  [{p['frame_auc_ci_lower']:.4f}, {p['frame_auc_ci_upper']:.4f}]")
    print(f"  EAR @ 0.5:       {p['ear']:.4f}")
    print(f"  MALT:            {p['malt_frames']:.1f} frames  ({p['malt_seconds']:.2f} s)")
    print(f"  Pre-crime AP:    {p['precrime_ap']:.4f}")
    print(f"  T-AUC@0:         {results['temporal_auc'].get('0', 0):.4f}")
    print(f"  T-AUC@30:        {results['temporal_auc'].get('30', 0):.4f}")
    print(f"  T-AUC@60:        {results['temporal_auc'].get('60', 0):.4f}")
    print(f"  T-AUC@90:        {results['temporal_auc'].get('90', 0):.4f}")
    print(f"  ECE (calib):     {results['calibration']['ece']:.4f}")
    print(f"{'='*55}\n")

    # Generate figures
    _generate_figures(results, score_arrays, gt_per_video, video_ids, Path(args.figures_dir))


if __name__ == "__main__":
    main()
