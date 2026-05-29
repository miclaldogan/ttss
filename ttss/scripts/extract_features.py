"""TTSS: offline feature extraction — YOLOv8m + ViT-B/16 per video.

Run this ONCE after downloading UCF-Crime videos.  Each video is processed
into a single .npz file containing:

    yolo_features  (T, 8)    float32 — YOLOv8m detection summary
    vit_features   (T, 768)  float32 — ViT-B/16 CLS embeddings
    frame_indices  (T,)      int32   — original frame indices sampled
    video_id       str
    label          str       e.g. 'Abuse', 'Normal_Videos'
    split          str       'train' | 'test'

Features are saved to:  data/features/{split}/{video_id}.npz

Usage::

    # Full extraction (takes ~10h on RTX 4050)
    python -m ttss.scripts.extract_features \\
        --videos-dir  data/raw/UCF-Crime/videos \\
        --output-dir  data/features \\
        --train-list  data/splits/ucf_crime_train.txt \\
        --test-list   data/splits/ucf_crime_test.txt \\
        --device      cuda

    # Quick smoke test (2 videos, no real data needed)
    python -m ttss.scripts.extract_features --smoke-test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_video_index(videos_dir: Path) -> dict[str, Path]:
    """Map video_id → Path for all video files under videos_dir."""
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    index: dict[str, Path] = {}
    for p in videos_dir.rglob("*"):
        if p.suffix.lower() in exts:
            stem = p.stem.replace("_x264", "")
            index[stem] = p
            index[p.stem] = p
    return index


def _load_video_frames(
    video_path: Path,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[list[int], list]:
    """Extract frames from a video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(range(0, total, frame_stride))
    if max_frames:
        indices = indices[:max_frames]
    selected = set(indices)
    frames, collected = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in selected:
            frames.append(frame)
            collected.append(idx)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return collected, frames


# ---------------------------------------------------------------------------
# Single-video extraction
# ---------------------------------------------------------------------------


def extract_video(
    video_path: Path,
    video_id: str,
    label: str,
    split: str,
    output_dir: Path,
    yolo_model,
    vit_encoder,
    frame_stride: int = 8,
    max_frames: int | None = None,
    device: str = "cuda",
    overwrite: bool = False,
) -> Path:
    """Extract and save features for one video."""
    out_path = output_dir / split / f"{video_id}.npz"
    if out_path.exists() and not overwrite:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices, frames = _load_video_frames(video_path, frame_stride, max_frames)
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    # YOLOv8m features
    yolo_feats = []
    for frame in frames:
        results = yolo_model.predict(source=frame, verbose=False, device=device)
        from ttss.models.recognition.yolov8_wrapper import YoloV8Wrapper
        # Re-use the wrapper's feature extraction
        wrapper = _get_yolo_wrapper(yolo_model, device)
        detections = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            from ttss.models.recognition.yolov8_wrapper import Detection
            for cls_id, conf, xyxy in zip(
                boxes.cls.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
                boxes.xyxy.detach().cpu().tolist(),
            ):
                raw_label = str(names.get(int(cls_id), cls_id))
                mapped = wrapper.normalize_label(raw_label)
                if mapped in wrapper.focus_classes:
                    detections.append(Detection(
                        label=mapped, confidence=float(conf),
                        xyxy=tuple(float(v) for v in xyxy),
                        frame_id=0, class_id=int(cls_id),
                    ))
        feat = wrapper.extract_feature_tensor(detections)
        yolo_feats.append(feat.numpy())

    # ViT-B/16 features — batch for efficiency
    import cv2
    vit_feats = []
    batch_size = 16
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_tensor = torch.stack([
            torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)
            for f in batch_frames
        ]).to(device) / 255.0
        with torch.no_grad():
            embeddings = vit_encoder.encode_batch(
                [batch_tensor[j] for j in range(len(batch_frames))]
            )
        vit_feats.append(embeddings.detach().cpu().numpy())

    yolo_arr = np.stack(yolo_feats).astype(np.float32)        # (T, 8)
    vit_arr = np.concatenate(vit_feats, axis=0).astype(np.float32)  # (T, 768)

    np.savez_compressed(
        out_path,
        yolo_features=yolo_arr,
        vit_features=vit_arr,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        video_id=video_id,
        label=label,
        split=split,
    )
    return out_path


# YoloV8Wrapper instance cache to avoid re-creating
_wrapper_cache: dict = {}


def _get_yolo_wrapper(yolo_model, device: str):
    if device not in _wrapper_cache:
        from ttss.models.recognition.yolov8_wrapper import YoloV8Wrapper
        w = YoloV8Wrapper(device=device)
        w.model = yolo_model  # reuse already-loaded model
        _wrapper_cache[device] = w
    return _wrapper_cache[device]


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def run_extraction(
    videos_dir: Path,
    output_dir: Path,
    train_list: Path,
    test_list: Path,
    device: str = "cuda",
    frame_stride: int = 8,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> None:
    from ultralytics import YOLO
    from ttss.models.detection.vit_scene import VitSceneEncoder

    print(f"Loading YOLOv8m...")
    yolo = YOLO("yolov8m.pt")

    print(f"Loading ViT-B/16...")
    vit = VitSceneEncoder(pretrained=True, device=device, num_unfreeze_blocks=0)
    vit.load()
    vit.eval()

    video_index = _build_video_index(videos_dir)
    print(f"Found {len(video_index)} videos under {videos_dir}")

    splits = {
        "train": (train_list, "train"),
        "test":  (test_list,  "test"),
    }

    total_done = total_skip = total_fail = 0
    t_start = time.perf_counter()

    for split_name, (list_path, split_tag) in splits.items():
        if not list_path.exists():
            print(f"  Skipping {split_name}: list file not found at {list_path}")
            continue
        video_ids = [l.strip() for l in list_path.read_text().splitlines() if l.strip()]
        print(f"\n{split_name.upper()}: {len(video_ids)} videos")

        for i, vid_id in enumerate(video_ids):
            # Infer category from video_id (e.g. Abuse001_x264 → Abuse)
            import re
            match = re.match(r'^([A-Za-z_]+?)(\d+)', vid_id.replace("_x264", ""))
            label = match.group(1).rstrip("_") if match else "Unknown"

            video_path = video_index.get(vid_id) or video_index.get(vid_id + "_x264")
            if video_path is None:
                total_fail += 1
                continue

            out_path = output_dir / split_tag / f"{vid_id}.npz"
            if out_path.exists() and not overwrite:
                total_skip += 1
                continue

            try:
                t0 = time.perf_counter()
                extract_video(
                    video_path, vid_id, label, split_tag,
                    output_dir, yolo, vit,
                    frame_stride=frame_stride, max_frames=max_frames,
                    device=device, overwrite=overwrite,
                )
                elapsed = time.perf_counter() - t0
                total_done += 1
                if (i + 1) % 10 == 0 or i == 0:
                    rate = total_done / (time.perf_counter() - t_start)
                    remaining = (len(video_ids) - i - 1) / max(rate, 1e-6)
                    print(
                        f"  [{i+1:4d}/{len(video_ids)}] {vid_id:<30} "
                        f"{elapsed:.1f}s  ETA {remaining/60:.0f}min"
                    )
            except Exception as exc:
                print(f"  FAIL {vid_id}: {exc}")
                total_fail += 1

    elapsed_total = time.perf_counter() - t_start
    print(f"\nDone — extracted={total_done}  skipped={total_skip}  failed={total_fail}  "
          f"time={elapsed_total/60:.1f}min")


# ---------------------------------------------------------------------------
# Smoke test (synthetic frames, no real data)
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    import tempfile, cv2, numpy as np
    print("Running smoke test (synthetic frames)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Fake a tiny video
        video_path = tmp / "test.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"XVID"), 30, (224, 224))
        for _ in range(30):
            writer.write(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        writer.release()

        from ultralytics import YOLO
        from ttss.models.detection.vit_scene import VitSceneEncoder
        yolo = YOLO("yolov8m.pt")
        vit = VitSceneEncoder(pretrained=True, device="cpu", num_unfreeze_blocks=0)
        vit.load(); vit.eval()

        out = extract_video(
            video_path, "smoke_test", "Robbery", "train",
            tmp / "features", yolo, vit,
            frame_stride=10, device="cpu",
        )

        data = np.load(out)
        print(f"  yolo_features: {data['yolo_features'].shape}")
        print(f"  vit_features:  {data['vit_features'].shape}")
        print(f"  video_id: {data['video_id']}")
        print("Smoke test PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTSS offline feature extraction")
    p.add_argument("--videos-dir", default="data/raw/UCF-Crime/videos")
    p.add_argument("--output-dir", default="data/features")
    p.add_argument("--train-list", default="data/splits/ucf_crime_train.txt")
    p.add_argument("--test-list",  default="data/splits/ucf_crime_test.txt")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame-stride", type=int, default=8)
    p.add_argument("--max-frames",   type=int, default=None)
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        run_smoke_test()
        return
    run_extraction(
        videos_dir=Path(args.videos_dir),
        output_dir=Path(args.output_dir),
        train_list=Path(args.train_list),
        test_list=Path(args.test_list),
        device=args.device,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
