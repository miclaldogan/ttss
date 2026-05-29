"""TTSS: inference efficiency benchmark — entry point with issue #15 CLI flags.

Usage::

    python -m ttss.scripts.benchmark_inference --dummy --device cpu
    python -m ttss.scripts.benchmark_inference --video path/to/video.mp4 --device cuda --warmup 10 --runs 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ttss.evaluation.efficiency import count_parameters, profile_forward


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTSS inference efficiency benchmark")
    p.add_argument("--video", default=None, help="Path to a real video file")
    p.add_argument("--dummy", action="store_true", help="Run with synthetic data (no video required)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--clip-length", type=int, default=64)
    p.add_argument("--output", default="evaluation/efficiency_report.json")
    return p


def _sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_component(fn, warmup, runs, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), float(np.percentile(arr, 95))


def main() -> None:
    args = build_parser().parse_args()
    if not args.dummy and args.video is None:
        print("Use --dummy to run without a real video, or provide --video path")
        return

    device, warmup, runs, T = args.device, args.warmup, args.runs, args.clip_length
    print(f"Device: {device.upper()}  |  warmup={warmup}  runs={runs}  clip_length={T}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []

    # YOLOv8m
    print("\nYOLOv8m ...", end=" ", flush=True)
    from ultralytics import YOLO
    yolo = YOLO("yolov8m.pt")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mean_ms, std_ms, p95_ms = _bench_component(
        lambda: yolo.predict(frame, verbose=False, device=device), warmup, runs, device)
    rows.append({"name": "YOLOv8m", "mean_ms": mean_ms, "std_ms": std_ms, "p95_ms": p95_ms,
                 "fps": 1000 / mean_ms})
    print(f"done — {mean_ms:.1f} ± {std_ms:.1f} ms/frame  p95={p95_ms:.1f} ms")

    # ViT-B/16
    print("ViT-B/16 (frozen) ...", end=" ", flush=True)
    from ttss.models.detection.vit_scene import VitSceneEncoder
    vit = VitSceneEncoder(pretrained=True, device=device, num_unfreeze_blocks=0)
    vit.load(); vit.eval()
    inp = torch.rand(1, 3, 224, 224, device=device)
    lat = profile_forward(vit, (1, 3, 224, 224), device=device, n_warmup=warmup, n_runs=runs)
    rows.append({"name": "ViT-B/16 (frozen)", "mean_ms": lat.mean_ms, "std_ms": lat.std_ms,
                 "p95_ms": lat.p95_ms, "fps": lat.fps})
    print(f"done — {lat.mean_ms:.1f} ms/frame  p95={lat.p95_ms:.1f} ms")

    # BiLSTM
    print(f"BiLSTM (T={T}) ...", end=" ", flush=True)
    from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor
    bilstm = BiLSTMThreatPredictor(input_dim=776).to(device).eval()
    lat_b = profile_forward(bilstm, (1, T, 776), device=device, n_warmup=warmup, n_runs=runs)
    rows.append({"name": f"BiLSTM T={T}", "mean_ms": lat_b.mean_ms / T,
                 "std_ms": lat_b.std_ms / T, "p95_ms": lat_b.p95_ms / T,
                 "fps": lat_b.fps * T})
    print(f"done — {lat_b.mean_ms / T:.3f} ms/frame  p95={lat_b.p95_ms / T:.3f} ms")

    # Parameter counts
    from ttss.models.end_to_end import EndToEndThreatModel
    e2e = EndToEndThreatModel.build(num_unfreeze_blocks=0, device=device)
    params = count_parameters(e2e)

    print(f"\n{'Component':<35} {'mean ms':>9} {'± std':>8} {'p95 ms':>8} {'FPS':>8}")
    print("-" * 74)
    for r in rows:
        print(f"{r['name']:<35} {r['mean_ms']:>9.2f} {r['std_ms']:>8.2f} "
              f"{r['p95_ms']:>8.2f} {r['fps']:>8.1f}")

    print(f"\nParameter counts: {json.dumps({k: v for k, v in params.items()}, indent=2)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"device": device, "warmup": warmup, "runs": runs,
                   "clip_length": T, "results": rows, "parameters": params}, f, indent=2)
    print(f"\nReport saved → {out}")


if __name__ == "__main__":
    main()
