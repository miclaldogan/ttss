"""Temporal Threat Scoring System (TTSS): inference efficiency benchmark.

Profiles each pipeline component in isolation and end-to-end, then prints a
formatted table of FPS, latency, and parameter counts.

Usage::

    python -m ttss.scripts.benchmark_efficiency [--device cuda] [--n-frames 100]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ttss.evaluation.efficiency import count_parameters, profile_forward, LatencyResult


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, n_warmup: int = 5, n_runs: int = 50, device: str = "cpu") -> float:
    """Return mean wall-clock time in milliseconds over *n_runs* calls."""
    for _ in range(n_warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n_runs * 1000.0


# ---------------------------------------------------------------------------
# Component benchmarks
# ---------------------------------------------------------------------------


def _bench_yolo(device: str, n_runs: int) -> dict:
    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    ms = _bench(lambda: model.predict(frame, verbose=False, device=device), n_runs=n_runs)
    params = sum(p.numel() for p in model.model.parameters())
    return {"name": "YOLOv8m", "ms_per_frame": ms, "fps": 1000 / ms, "params_M": params / 1e6}


def _bench_vit(device: str, n_runs: int) -> dict:
    from ttss.models.detection.vit_scene import VitSceneEncoder
    enc = VitSceneEncoder(pretrained=True, device=device, num_unfreeze_blocks=0)
    enc.load()
    enc.eval()
    frame = torch.rand(1, 3, 224, 224, device=device)
    total_p = sum(p.numel() for p in enc.model.parameters())
    trainable_p = sum(p.numel() for p in enc.model.parameters() if p.requires_grad)
    with torch.no_grad():
        ms = _bench(lambda: enc.forward(frame), n_runs=n_runs, device=device)
    return {
        "name": "ViT-B/16 (frozen)",
        "ms_per_frame": ms,
        "fps": 1000 / ms,
        "params_M": total_p / 1e6,
        "trainable_params_M": trainable_p / 1e6,
    }


def _bench_vit_finetune(device: str, n_runs: int) -> dict:
    from ttss.models.detection.vit_scene import VitSceneEncoder
    enc = VitSceneEncoder(pretrained=True, device=device, num_unfreeze_blocks=2)
    enc.load()
    enc.train()
    frame = torch.rand(1, 3, 224, 224, device=device)
    trainable_p = sum(p.numel() for p in enc.model.parameters() if p.requires_grad)
    # Include backward pass timing
    def _step():
        out = enc.forward(frame)
        out.mean().backward()
        enc.zero_grad()
    ms = _bench(_step, n_runs=n_runs, device=device)
    return {
        "name": "ViT-B/16 (last-2 blocks, fwd+bwd)",
        "ms_per_frame": ms,
        "fps": 1000 / ms,
        "trainable_params_M": trainable_p / 1e6,
    }


def _bench_bilstm(device: str, n_runs: int, T: int = 64) -> dict:
    from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor
    model = BiLSTMThreatPredictor(input_dim=776).to(device).eval()
    x = torch.rand(1, T, 776, device=device)
    lat = profile_forward(model, (1, T, 776), device=device, n_warmup=5, n_runs=n_runs, name=f"BiLSTM T={T}")
    params = count_parameters(model)
    return {
        "name": f"BiLSTM (T={T})",
        "ms_per_clip": lat.mean_ms,
        "ms_per_frame": lat.mean_ms / T,
        "p95_ms": lat.p95_ms / T,
        "fps": lat.fps * T,
        "params_M": params["total"] / 1e6,
    }


def _bench_end_to_end(device: str, n_runs: int, T: int = 64) -> dict:
    """Profile full pipeline: YOLO (pre-computed) + ViT (frozen) + BiLSTM."""
    from ttss.models.end_to_end import EndToEndThreatModel
    model = EndToEndThreatModel.build(num_unfreeze_blocks=0, device=device)
    model.eval()
    frames = torch.rand(1, T, 3, 224, 224, device=device)
    yolo = torch.rand(1, T, 8, device=device)
    total_p = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        ms = _bench(lambda: model((frames, yolo)), n_runs=n_runs, device=device)
    return {
        "name": f"End-to-end ViT+BiLSTM (T={T}, YOLO pre-comp)",
        "ms_per_clip": ms,
        "ms_per_frame": ms / T,
        "fps": T * 1000 / ms,
        "params_M": total_p / 1e6,
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict]) -> None:
    print()
    print(f"{'Component':<45} {'mean ms':>9} {'p95 ms':>8} {'FPS':>8} {'Params (M)':>12}")
    print("-" * 86)
    for r in rows:
        ms = r.get("ms_per_frame", r.get("ms_per_clip", 0.0))
        p95 = r.get("p95_ms_per_frame", r.get("p95_ms", ms))
        fps = r.get("fps", 0.0)
        params = r.get("params_M", r.get("trainable_params_M", 0.0))
        print(f"{r['name']:<45} {ms:>9.2f} {p95:>8.2f} {fps:>8.1f} {params:>12.2f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTSS inference efficiency benchmark")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--clip-length", type=int, default=64)
    p.add_argument("--output", default=None, help="Save results JSON to this path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device, n_runs, T = args.device, args.n_runs, args.clip_length

    print(f"Benchmarking on {device.upper()}  |  n_runs={n_runs}  |  clip_length={T}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    print("\nRunning YOLOv8m ...", end=" ", flush=True)
    rows.append(_bench_yolo(device, n_runs)); print("done")

    print("Running ViT-B/16 (frozen) ...", end=" ", flush=True)
    rows.append(_bench_vit(device, n_runs)); print("done")

    print("Running ViT-B/16 fine-tune (fwd+bwd) ...", end=" ", flush=True)
    rows.append(_bench_vit_finetune(device, n_runs)); print("done")

    print(f"Running BiLSTM (T={T}) ...", end=" ", flush=True)
    rows.append(_bench_bilstm(device, n_runs, T)); print("done")

    print(f"Running end-to-end (T={T}) ...", end=" ", flush=True)
    rows.append(_bench_end_to_end(device, n_runs, T)); print("done")

    _print_table(rows)

    report_path = args.output or "evaluation/efficiency_report.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"device": device, "n_runs": n_runs, "clip_length": T, "results": rows}, f, indent=2)
    print(f"Results saved → {report_path}")


if __name__ == "__main__":
    main()
