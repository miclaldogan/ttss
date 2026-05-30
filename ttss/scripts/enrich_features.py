"""TTSS: incremental feature enrichment.

Augments existing .npz feature files WITHOUT re-running VideoMAE (~4h saved).
For each video the script:
  1. Reads the video once sequentially to collect all clips.
  2. Gathers the center frame from every clip into one list and runs YOLOv8m in
     a single batched call → fast GPU utilisation (~10-30 min for 1900 videos).
  3. Extracts 24 rich spatial/interaction features per detection result and
     stacks them with the original 8-dim summary → 32-dim YOLO feature vector.
  4. Computes fast frame-difference motion statistics for every clip → 16-dim.
  5. Overwrites the .npz with updated yolo_features (T,32) + flow_features (T,16)
     while leaving vit_features (T,768) untouched.

Rich YOLO feature layout (32-dim per clip)
    Inherited from original extractor (8-dim):
     0  person_count
     1  car_count
     2  weapon_count
     3  mean_confidence (all dets)
     4  max_confidence
     5  mean_area
     6  person_confidence_sum
     7  weapon_confidence_sum
    New spatial/interaction features (24-dim):
     8  person_cx_mean           normalised [0,1]
     9  person_cy_mean
    10  person_spatial_spread    std(cx) + std(cy)
    11  person_min_pairwise_dist normalised by frame diagonal
    12  person_mean_pairwise_dist
    13  person_mean_area
    14  person_area_std
    15  person_bottom_y_mean     bottom edge of bbox / frame height
    16  person_aspect_ratio_mean w/h
    17  total_bbox_occupancy     sum of all bbox areas / frame area
    18  has_weapon               binary
    19  crowd_flag               person_count >= 3
    20  weapon_near_person       any weapon bbox overlaps a person bbox
    21  crowding_score           person_count / (1 + min_pairwise_dist)
    22  person_top_half_ratio    fraction of persons whose centre is in top half
    23  max_person_confidence    max conf among person detections
    24  n_total_detections       total number of detections in frame
    25  detection_conf_std       std of all detection confidences
    26  car_max_area             largest vehicle bbox area
    27  person_bbox_density      person_count / (convex-hull-area + 1e-4)
    28  person_x_spread          max(cx) - min(cx) among persons
    29  person_y_spread          max(cy) - min(cy) among persons
    30  frame_occupancy_person   total person bbox area / frame area
    31  max_area_any             largest detection bbox area

Motion feature layout (16-dim per clip, from frame differences)
     0  motion_mean              mean |Δ| over all consecutive frame pairs (÷255)
     1  motion_std
     2  motion_q75               75th-pct of |Δ| values
     3  motion_q95               95th-pct
     4  motion_coverage          fraction of pixels with |Δ|>10
     5..13  spatial_3x3          mean |Δ| in each cell of a 3×3 grid (÷255)
    14  early_motion             mean clip-level motion in first half of clip
    15  late_motion              mean clip-level motion in second half

Usage::

    python -m ttss.scripts.enrich_features --device cuda

    # Force re-enrichment even if already done
    python -m ttss.scripts.enrich_features --device cuda --overwrite
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CLIP_LEN = 16
MAX_CLIPS = 200           # same cap as original extractor
YOLO_BATCH = 64           # number of frames per batched YOLO call
YOLO_RICH_DIM = 32
FLOW_DIM = 16


# ---------------------------------------------------------------------------
# Rich YOLO per-frame feature (32-dim)
# ---------------------------------------------------------------------------


def _extract_rich_yolo(result: Any) -> np.ndarray:
    """Convert a single YOLOv8 result object → 32-dim rich feature vector."""
    feat = np.zeros(YOLO_RICH_DIM, dtype=np.float32)

    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)

    persons: list[dict] = []
    cars:    list[dict] = []
    weapons: list[dict] = []
    all_dets: list[dict] = []

    if boxes is not None and len(boxes):
        orig_shape = result.orig_shape          # (H, W)
        h, w = float(orig_shape[0]), float(orig_shape[1])

        for cls_id, conf, xyxy in zip(
            boxes.cls.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.xyxy.cpu().tolist(),
        ):
            label = str(names.get(int(cls_id), "")).lower()
            x1, y1, x2, y2 = xyxy
            cx   = (x1 + x2) / (2.0 * w)
            cy   = (y1 + y2) / (2.0 * h)
            area = (x2 - x1) * (y2 - y1) / (w * h + 1e-9)
            det  = dict(
                conf=float(conf), cx=cx, cy=cy, area=area,
                aspect=(x2 - x1) / max(y2 - y1, 1.0),
                bottom_y=y2 / h, top_y=y1 / h,
                x1=x1 / w, y1=y1 / h, x2=x2 / w, y2=y2 / h,
            )
            all_dets.append(det)
            if label == "person":
                persons.append(det)
            elif label in ("car", "bus", "truck", "motorcycle", "bicycle"):
                cars.append(det)
            elif (label in ("knife", "scissors", "baseball bat")
                  or "gun" in label or "weapon" in label or "rifle" in label):
                weapons.append(det)

    # ---- Original 8 features ----
    conf_vals = [d["conf"] for d in all_dets]
    area_vals = [d["area"] for d in all_dets]
    feat[0] = float(len(persons))
    feat[1] = float(len(cars))
    feat[2] = float(len(weapons))
    feat[3] = float(np.mean(conf_vals))   if conf_vals else 0.0
    feat[4] = float(max(conf_vals))       if conf_vals else 0.0
    feat[5] = float(np.mean(area_vals))   if area_vals else 0.0
    feat[6] = float(sum(d["conf"] for d in persons))
    feat[7] = float(sum(d["conf"] for d in weapons))

    # ---- New spatial/interaction features ----
    if persons:
        cxs  = [p["cx"]       for p in persons]
        cys  = [p["cy"]       for p in persons]
        arrs = [p["area"]     for p in persons]
        bots = [p["bottom_y"] for p in persons]
        tops = [p["top_y"]    for p in persons]
        asps = [p["aspect"]   for p in persons]

        feat[8]  = float(np.mean(cxs))
        feat[9]  = float(np.mean(cys))
        feat[10] = float(np.std(cxs) + np.std(cys))
        feat[13] = float(np.mean(arrs))
        feat[14] = float(np.std(arrs))
        feat[15] = float(np.mean(bots))
        feat[16] = float(np.mean(asps))
        feat[22] = float(np.mean([1.0 if cy < 0.5 else 0.0 for cy in cys]))
        feat[23] = float(max(p["conf"] for p in persons))
        feat[28] = float(max(cxs) - min(cxs)) if len(cxs) > 1 else 0.0
        feat[29] = float(max(cys) - min(cys)) if len(cys) > 1 else 0.0
        feat[30] = float(sum(arrs))

        if len(persons) >= 2:
            dists = [
                ((p1["cx"] - p2["cx"]) ** 2 + (p1["cy"] - p2["cy"]) ** 2) ** 0.5
                for p1, p2 in combinations(persons, 2)
            ]
            feat[11] = float(min(dists))
            feat[12] = float(np.mean(dists))
        else:
            feat[11] = feat[12] = 1.0

        # Convex-hull area proxy (axis-aligned bounding box of all persons)
        hull_area = (max(cxs) - min(cxs)) * (max(cys) - min(cys)) if len(persons) > 1 else 1e-4
        feat[27] = float(len(persons)) / (hull_area + 1e-4)
        feat[21] = float(len(persons)) / (1.0 + feat[11])
    else:
        feat[8] = feat[9] = 0.5
        feat[11] = feat[12] = 1.0

    feat[17] = float(sum(area_vals))
    feat[18] = 1.0 if weapons else 0.0
    feat[19] = 1.0 if len(persons) >= 3 else 0.0

    # Weapon-person bbox overlap
    if weapons and persons:
        for wep in weapons:
            for per in persons:
                if not (wep["x2"] < per["x1"] or per["x2"] < wep["x1"]
                        or wep["y2"] < per["y1"] or per["y2"] < wep["y1"]):
                    feat[20] = 1.0
                    break

    feat[24] = float(len(all_dets))
    feat[25] = float(np.std(conf_vals)) if len(conf_vals) > 1 else 0.0
    feat[26] = float(max(d["area"] for d in cars)) if cars else 0.0
    feat[31] = float(max(area_vals)) if area_vals else 0.0

    return feat


# ---------------------------------------------------------------------------
# Frame-difference motion features (16-dim)  — much faster than Farneback
# ---------------------------------------------------------------------------


def _compute_motion_features(frames: list) -> np.ndarray:
    """Compute 16-dim frame-difference motion statistics from a clip."""
    out = np.zeros(FLOW_DIM, dtype=np.float32)
    if len(frames) < 2:
        return out

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]

    diffs_temporal: list[float] = []          # per-pair mean absolute difference
    all_flat: list[np.ndarray] = []

    for i in range(len(grays) - 1):
        d = np.abs(grays[i + 1] - grays[i])
        diffs_temporal.append(float(d.mean()))
        all_flat.append(d.flatten())

    flat = np.concatenate(all_flat)

    out[0] = float(np.mean(flat)) / 255.0
    out[1] = float(np.std(flat))  / 255.0
    out[2] = float(np.percentile(flat, 75))  / 255.0
    out[3] = float(np.percentile(flat, 95))  / 255.0
    out[4] = float(np.mean(flat > 10.0))

    # Temporal mean motion heatmap for spatial decomposition
    mean_diff = np.mean([np.abs(grays[i + 1] - grays[i]) for i in range(len(grays) - 1)], axis=0)
    H, W = mean_diff.shape
    h3, w3 = max(1, H // 3), max(1, W // 3)
    idx = 5
    for row in range(3):
        for col in range(3):
            r0, r1 = row * h3, (row + 1) * h3 if row < 2 else H
            c0, c1 = col * w3, (col + 1) * w3 if col < 2 else W
            out[idx] = float(mean_diff[r0:r1, c0:c1].mean()) / 255.0
            idx += 1   # fills [5..13]

    n = len(diffs_temporal)
    mid = n // 2
    out[14] = float(np.mean(diffs_temporal[:mid]))   / 255.0 if mid > 0 else 0.0
    out[15] = float(np.mean(diffs_temporal[mid:]))   / 255.0 if mid < n else 0.0

    return out


# ---------------------------------------------------------------------------
# Per-video enrichment (sequential read + batched YOLO)
# ---------------------------------------------------------------------------


def enrich_video(
    npz_path: Path,
    video_path: Path,
    yolo_model,
    device: str,
) -> None:
    """Read video once, compute rich YOLO (32-dim) + motion (16-dim), overwrite npz."""
    data = dict(np.load(npz_path, allow_pickle=True))
    frame_indices = data["frame_indices"]       # (T,) int32 — center frames
    T_target = min(int(len(frame_indices)), MAX_CLIPS)

    # ------------------------------------------------------------------
    # Step 1: Stream all clips sequentially (no seeks) and collect frames
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    center_frames: list  = []   # one center frame per clip (for YOLO)
    clip_frames_all: list = []  # full 16-frame clips (for motion)

    clip_count = 0
    while clip_count < MAX_CLIPS:
        clip = []
        for _ in range(CLIP_LEN):
            ok, f = cap.read()
            if not ok:
                if clip:
                    clip.append(clip[-1].copy())
                break
            clip.append(f)
        if len(clip) < CLIP_LEN:
            break
        center_frames.append(clip[CLIP_LEN // 2])
        clip_frames_all.append(clip)
        clip_count += 1

    cap.release()

    # Align to the number of clips stored in the npz.
    # If the video is shorter than expected, pad with zeros to stay consistent
    # with frame_indices length so downstream loading sees matching shapes.
    T_actual = min(T_target, len(center_frames))

    # ------------------------------------------------------------------
    # Step 2: Batched YOLO inference on all center frames at once
    # ------------------------------------------------------------------
    yolo_rich  = np.zeros((T_target, YOLO_RICH_DIM), dtype=np.float32)
    flow_feats = np.zeros((T_target, FLOW_DIM),       dtype=np.float32)
    T = T_actual  # only process real clips; remaining rows stay as zeros

    all_results: list = []
    for start in range(0, T, YOLO_BATCH):
        batch = center_frames[start: start + YOLO_BATCH]
        results = yolo_model.predict(
            source=batch, verbose=False, device=device, batch=len(batch)
        )
        all_results.extend(results)

    # ------------------------------------------------------------------
    # Step 3: Extract per-clip features
    # ------------------------------------------------------------------
    for t in range(T):
        yolo_rich[t]  = _extract_rich_yolo(all_results[t])
        flow_feats[t] = _compute_motion_features(clip_frames_all[t])

    # ------------------------------------------------------------------
    # Step 4: Overwrite npz  (vit_features, frame_indices, etc. unchanged)
    # ------------------------------------------------------------------
    data["yolo_features"] = yolo_rich
    data["flow_features"] = flow_feats
    np.savez_compressed(npz_path, **{k: v for k, v in data.items()})


# ---------------------------------------------------------------------------
# Video index helper
# ---------------------------------------------------------------------------


def _build_video_index(videos_dir: Path) -> dict[str, Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    index: dict[str, Path] = {}
    for p in videos_dir.rglob("*"):
        if p.suffix.lower() in exts:
            index[p.stem] = p
            index[p.stem.replace("_x264", "")] = p
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich existing .npz feature files with richer YOLO (32-dim) "
                    "and motion (16-dim) features without re-running VideoMAE."
    )
    parser.add_argument("--features-dir", default="data/features",
                        help="Root of .npz feature files (train/ and test/ subdirs)")
    parser.add_argument("--videos-dir",   default="data/raw/UCF-Crime/videos")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-enrich even if flow_features already present with correct dims")
    args = parser.parse_args()

    from ultralytics import YOLO
    print("Loading YOLOv8m...")
    yolo = YOLO("yolov8m.pt")

    features_root = Path(args.features_dir)
    videos_root   = Path(args.videos_dir)

    video_index = _build_video_index(videos_root)
    print(f"Found {len(video_index)} videos under {videos_root}")

    npz_files = sorted(features_root.rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files to process\n")

    done = skip = fail = 0
    t_start = time.perf_counter()

    for i, npz_path in enumerate(npz_files):
        video_id = npz_path.stem

        if not args.overwrite:
            d = np.load(npz_path, allow_pickle=True)
            if (
                "flow_features" in d.files
                and d["yolo_features"].shape[1] == YOLO_RICH_DIM
            ):
                skip += 1
                continue

        video_path = video_index.get(video_id) or video_index.get(video_id + "_x264")
        if video_path is None:
            print(f"  SKIP (no video): {video_id}")
            fail += 1
            continue

        try:
            t0 = time.perf_counter()
            enrich_video(npz_path, video_path, yolo, args.device)
            elapsed = time.perf_counter() - t0
            done += 1
            if done % 50 == 0 or done <= 3:
                rate = done / (time.perf_counter() - t_start)
                eta  = (len(npz_files) - i - 1) / max(rate, 1e-9)
                print(
                    f"  [{i+1:4d}/{len(npz_files)}] {video_id:<38} "
                    f"{elapsed:5.1f}s  rate={rate:.2f}v/s  ETA {eta/60:.0f}min"
                )
        except Exception as exc:
            import traceback
            print(f"  FAIL {video_id}: {exc}")
            traceback.print_exc()
            fail += 1

    elapsed_total = time.perf_counter() - t_start
    print(
        f"\nDone — enriched={done}  skipped={skip}  failed={fail}  "
        f"time={elapsed_total / 60:.1f}min"
    )


if __name__ == "__main__":
    main()
