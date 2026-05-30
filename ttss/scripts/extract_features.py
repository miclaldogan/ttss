"""TTSS: offline feature extraction — YOLOv8m + VideoMAE-Base per video.

Run this ONCE after downloading UCF-Crime videos.  Each video is processed
into a single .npz file containing:

    yolo_features  (T, 8)    float32 — YOLOv8m detection summary per clip
    vit_features   (T, 768)  float32 — VideoMAE-Base clip embeddings
    frame_indices  (T,)      int32   — center frame index of each clip
    video_id       str
    label          str       e.g. 'Abuse', 'Normal_Videos'
    split          str       'train' | 'test'

VideoMAE processes 16-frame clips (non-overlapping, stride=16).  Each clip
yields one 768-dim embedding via mean-pooling patch tokens — capturing motion
across frames rather than per-frame appearance only.

Features are saved to:  data/features/{split}/{video_id}.npz

Usage::

    # Full extraction
    python -m ttss.scripts.extract_features --device cuda

    # Quick smoke test (2 videos, no real data needed)
    python -m ttss.scripts.extract_features --smoke-test
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch


CLIP_LEN = 16       # frames per VideoMAE clip
CLIP_STRIDE = 16    # non-overlapping clips
VMAE_BATCH = 12     # clips per VideoMAE forward pass (1.1 GB used → can push to 12)
MAX_CLIPS = 200     # cap very long videos to keep RAM and time bounded


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_videomae(device: str):
    from transformers import VideoMAEModel, AutoImageProcessor
    print("Loading VideoMAE-Base (MCG-NJU/videomae-base)...")
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    model = model.to(device).eval()
    return processor, model


def _load_yolo(device: str):
    from ultralytics import YOLO
    print("Loading YOLOv8m...")
    return YOLO("yolov8m.pt")


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def _stream_clips(video_path: Path, clip_len: int, clip_stride: int, max_clips: int):
    """Stream non-overlapping clips from a video without loading all frames into RAM.

    Yields (clip_start_frame, center_frame_idx, list_of_clip_len_frames).
    clip_stride must equal clip_len (non-overlapping) for sequential streaming.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    clip_num = 0
    abs_frame = 0

    while clip_num < max_clips:
        clip_frames = []
        for _ in range(clip_len):
            ok, frame = cap.read()
            if not ok:
                break
            clip_frames.append(frame)
            abs_frame += 1

        if len(clip_frames) < clip_len:
            break  # end of video

        clip_start = clip_num * clip_stride
        center = clip_start + clip_len // 2
        yield clip_start, center, clip_frames
        clip_num += 1

        # Skip frames between clips if stride > clip_len
        for _ in range(clip_stride - clip_len):
            ok, _ = cap.read()
            if not ok:
                break
            abs_frame += 1

    cap.release()


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


_yolo_wrapper_cache: dict = {}


def _yolo_feature(yolo_model, frame, device: str) -> np.ndarray:
    """Run YOLOv8m on one frame, return (8,) detection summary."""
    from ttss.models.recognition.yolov8_wrapper import YoloV8Wrapper, Detection
    if device not in _yolo_wrapper_cache:
        w = YoloV8Wrapper(device=device)
        w.model = yolo_model
        _yolo_wrapper_cache[device] = w
    wrapper = _yolo_wrapper_cache[device]
    results = yolo_model.predict(source=frame, verbose=False, device=device)
    names = getattr(results[0], "names", {}) if results else {}
    boxes = getattr(results[0], "boxes", None) if results else None
    detections = []
    if boxes is not None:
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
    return wrapper.extract_feature_tensor(detections).numpy()


def _videomae_batch(
    clips: list[list],
    processor,
    model,
    device: str,
) -> np.ndarray:
    """Run VideoMAE on a batch of clips. clips: list of [16 BGR frames].
    Returns (N, 768) float32."""
    import cv2
    # Convert BGR→RGB for each clip
    rgb_clips = [
        [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip]
        for clip in clips
    ]
    # Processor expects a flat list of frames when num_frames is set
    # For batching: process each clip separately then stack
    pixel_values_list = []
    for rgb_frames in rgb_clips:
        inputs = processor(rgb_frames, return_tensors="pt")
        pixel_values_list.append(inputs["pixel_values"])  # (1, 16, 3, 224, 224)
    pixel_values = torch.cat(pixel_values_list, dim=0).to(device)  # (N, 16, 3, 224, 224)

    with torch.no_grad():
        with torch.autocast("cuda", torch.float16, enabled=(device == "cuda")):
            outputs = model(pixel_values=pixel_values)
    # Mean-pool patch tokens → (N, 768)
    embeddings = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
    return embeddings


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
    vmae_processor,
    vmae_model,
    device: str = "cuda",
    overwrite: bool = False,
) -> Path:
    """Extract and save VideoMAE + YOLO features for one video."""
    out_path = output_dir / split / f"{video_id}.npz"
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream clips directly — never load full video into RAM
    clip_batch: list = []
    center_batch: list = []
    yolo_feats_list: list = []
    vmae_feats_list: list = []

    first_clip = True
    for clip_start, center, clip_frames in _stream_clips(
        video_path, CLIP_LEN, CLIP_STRIDE, MAX_CLIPS
    ):
        clip_batch.append(clip_frames)
        center_batch.append(center)

        if len(clip_batch) == VMAE_BATCH:
            # YOLO on center frame of each clip in batch
            for c in clip_batch:
                yolo_feats_list.append(_yolo_feature(yolo_model, c[CLIP_LEN // 2], device))
            vmae_feats_list.append(
                _videomae_batch(clip_batch, vmae_processor, vmae_model, device)
            )
            clip_batch = []
            first_clip = False

    # Flush remaining clips
    if clip_batch:
        if first_clip:
            # Video too short — pad last clip
            while len(clip_batch[0]) < CLIP_LEN:
                clip_batch[0].append(clip_batch[0][-1])
        for c in clip_batch:
            yolo_feats_list.append(_yolo_feature(yolo_model, c[CLIP_LEN // 2], device))
        vmae_feats_list.append(
            _videomae_batch(clip_batch, vmae_processor, vmae_model, device)
        )

    if not yolo_feats_list:
        raise RuntimeError(f"No clips extracted from {video_path}")

    center_indices = center_batch
    yolo_feats = np.stack(yolo_feats_list)                                      # (T, 8)
    vmae_feats = np.concatenate(vmae_feats_list, axis=0).astype(np.float32)    # (T, 768)

    np.savez_compressed(
        out_path,
        yolo_features=yolo_feats.astype(np.float32),
        vit_features=vmae_feats,      # key kept for downstream compatibility
        frame_indices=np.array(center_indices, dtype=np.int32),
        video_id=video_id,
        label=label,
        split=split,
    )
    return out_path


# ---------------------------------------------------------------------------
# Video index
# ---------------------------------------------------------------------------


def _build_video_index(videos_dir: Path) -> dict[str, Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    index: dict[str, Path] = {}
    for p in videos_dir.rglob("*"):
        if p.suffix.lower() in exts:
            stem = p.stem.replace("_x264", "")
            index[stem] = p
            index[p.stem] = p
    return index


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def run_extraction(
    videos_dir: Path,
    output_dir: Path,
    train_list: Path,
    test_list: Path,
    device: str = "cuda",
    overwrite: bool = False,
) -> None:
    yolo = _load_yolo(device)
    vmae_processor, vmae_model = _load_videomae(device)

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
            print(f"  Skipping {split_name}: list not found at {list_path}")
            continue
        video_ids = [l.strip() for l in list_path.read_text().splitlines() if l.strip()]
        print(f"\n{split_name.upper()}: {len(video_ids)} videos")

        for i, vid_id in enumerate(video_ids):
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
                    output_dir, yolo, vmae_processor, vmae_model,
                    device=device, overwrite=overwrite,
                )
                elapsed = time.perf_counter() - t0
                total_done += 1
                if (i + 1) % 10 == 0 or i == 0:
                    rate = total_done / (time.perf_counter() - t_start)
                    remaining = (len(video_ids) - i - 1) / max(rate, 1e-6)
                    print(
                        f"  [{i+1:4d}/{len(video_ids)}] {vid_id:<35} "
                        f"{elapsed:.1f}s  rate={rate:.2f}vid/s  ETA {remaining/60:.0f}min"
                    )
            except Exception as exc:
                print(f"  FAIL {vid_id}: {exc}")
                total_fail += 1

    elapsed_total = time.perf_counter() - t_start
    print(f"\nDone — extracted={total_done}  skipped={total_skip}  failed={total_fail}  "
          f"time={elapsed_total/60:.1f}min")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    import tempfile, cv2
    print("Running smoke test (synthetic frames, VideoMAE-Base)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video_path = tmp / "test.avi"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"XVID"), 30, (224, 224)
        )
        for _ in range(48):
            writer.write(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        writer.release()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        yolo = _load_yolo(device)
        processor, model = _load_videomae(device)

        out = extract_video(
            video_path, "smoke_test", "Robbery", "train",
            tmp / "features", yolo, processor, model,
            device=device,
        )
        data = np.load(out)
        print(f"  yolo_features : {data['yolo_features'].shape}")
        print(f"  vit_features  : {data['vit_features'].shape}")
        print(f"  video_id      : {data['video_id']}")
        print("Smoke test PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTSS offline feature extraction (VideoMAE-Base)")
    p.add_argument("--videos-dir",  default="data/raw/UCF-Crime/videos")
    p.add_argument("--output-dir",  default="data/features")
    p.add_argument("--train-list",  default="data/splits/ucf_crime_train.txt")
    p.add_argument("--test-list",   default="data/splits/ucf_crime_test.txt")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--smoke-test",  action="store_true")
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
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
