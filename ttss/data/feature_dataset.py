"""TTSS: PyTorch Dataset over pre-extracted YOLOv8m + ViT feature files.

Each .npz file (written by extract_features.py) contains:
    yolo_features  (T_raw, 8)    — per-frame YOLO detection features
    vit_features   (T_raw, 768)  — per-frame ViT CLS embeddings
    frame_indices  (T_raw,)      — original frame indices
    video_id, label, split

This dataset:
  1. Loads a .npz file, concatenates YOLO + ViT → (T_raw, 776) feature matrix
  2. Clips/pads to a fixed clip_length T
  3. Derives a weak video-level label (1 = anomaly, 0 = normal)
  4. Optionally derives per-frame pseudo threat scores from temporal labeler
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


NORMAL_CATEGORIES = {"Normal_Videos", "Normal", "normal"}


@dataclass
class FeatureSample:
    """One clip of pre-extracted features."""

    features: torch.Tensor      # (T, yolo_dim+768) float32  — concatenated yolo+vmae
    yolo: torch.Tensor          # (T, yolo_dim) float32  — YOLO object features
    vit: torch.Tensor           # (T, 768) float32  — VideoMAE clip embeddings
    flow: torch.Tensor          # (T, 16)  float32  — optical-flow features (zeros if absent)
    labels: torch.Tensor        # (T,) float32  — 0.0 or 1.0 (weak)
    frame_indices: np.ndarray   # (T,) int32 — original video frame numbers
    video_id: str
    category: str
    is_anomaly: bool
    n_frames: int               # actual frames (before padding)


class FeatureDataset(Dataset):
    """Read pre-extracted feature .npz files and yield fixed-length clips.

    Parameters
    ----------
    feature_dir:   Directory of .npz files (e.g. data/features/train/).
    clip_length:   Fixed temporal length T.  Longer sequences are truncated;
                   shorter ones are zero-padded.
    split:         'train' | 'test' | None (load all).
    categories:    Whitelist of category names; None = all categories.
    """

    def __init__(
        self,
        feature_dir: str | Path,
        clip_length: int = 64,
        split: str | None = None,
        categories: Sequence[str] | None = None,
    ) -> None:
        self.feature_dir = Path(feature_dir)
        self.clip_length = clip_length
        self.split = split
        self.categories = set(categories) if categories else None

        self._files: list[Path] = []
        search_dirs = [self.feature_dir / split] if split else [
            self.feature_dir / "train",
            self.feature_dir / "test",
            self.feature_dir,
        ]
        for d in search_dirs:
            if d.exists():
                self._files.extend(sorted(d.glob("*.npz")))

        if self.categories is not None:
            self._files = [
                f for f in self._files
                if _infer_category(f.stem) in self.categories
            ]

        # Preload all .npz files into RAM to avoid repeated disk I/O each epoch
        print(f"  Preloading {len(self._files)} feature files into RAM...", flush=True)
        self._cache: list[tuple] = []
        for f in self._files:
            d = np.load(f, allow_pickle=True)
            yolo_raw = d["yolo_features"].astype(np.float32)
            T_raw = len(yolo_raw)
            # Pad unenriched (8-dim) YOLO features to 32-dim with zeros
            if yolo_raw.shape[1] < 32:
                yolo_raw = np.pad(yolo_raw, ((0, 0), (0, 32 - yolo_raw.shape[1])))
            # flow_features may be absent in pre-enrichment files — fall back to zeros
            if "flow_features" in d.files:
                flow_raw = d["flow_features"].astype(np.float32)
            else:
                flow_raw = np.zeros((T_raw, 16), dtype=np.float32)
            self._cache.append((
                yolo_raw,
                d["vit_features"].astype(np.float32),
                flow_raw,
                d["frame_indices"].astype(np.int32),
                str(d["video_id"]),
                str(d["label"]),
            ))
        print(f"  Done preloading.", flush=True)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> FeatureSample:
        yolo_raw, vit_raw, flow_raw, frame_indices_raw, video_id, label = self._cache[idx]
        features_raw = np.concatenate([yolo_raw, vit_raw], axis=1)

        is_anomaly = label not in NORMAL_CATEGORIES

        T_raw = len(features_raw)
        features, n_frames = _clip_and_pad(features_raw, self.clip_length)
        yolo, _            = _clip_and_pad(yolo_raw,     self.clip_length)
        vit,  _            = _clip_and_pad(vit_raw,      self.clip_length)
        flow, _            = _clip_and_pad(flow_raw,     self.clip_length)

        # Clip/pad frame_indices the same way (pad with -1 to mark padding)
        if T_raw >= self.clip_length:
            frame_indices = frame_indices_raw[:self.clip_length].copy()
        else:
            pad = np.full(self.clip_length - T_raw, -1, dtype=np.int32)
            frame_indices = np.concatenate([frame_indices_raw, pad])

        label_val = 1.0 if is_anomaly else 0.0
        labels = torch.full((self.clip_length,), label_val, dtype=torch.float32)

        return FeatureSample(
            features=features,
            yolo=yolo,
            vit=vit,
            flow=flow,
            labels=labels,
            frame_indices=frame_indices,
            video_id=video_id,
            category=label,
            is_anomaly=is_anomaly,
            n_frames=n_frames,
        )

    def anomaly_indices(self) -> list[int]:
        """Return indices of all anomaly samples."""
        return [i for i, f in enumerate(self._files)
                if _infer_category(f.stem) not in NORMAL_CATEGORIES]

    def normal_indices(self) -> list[int]:
        """Return indices of all normal samples."""
        return [i for i, f in enumerate(self._files)
                if _infer_category(f.stem) in NORMAL_CATEGORIES]


def _clip_and_pad(
    features: np.ndarray,
    clip_length: int,
) -> tuple[torch.Tensor, int]:
    """Truncate or zero-pad feature sequence to clip_length."""
    T = len(features)
    if T >= clip_length:
        clipped = features[:clip_length]
        n_frames = clip_length
    else:
        pad = np.zeros((clip_length - T, features.shape[1]), dtype=np.float32)
        clipped = np.concatenate([features, pad], axis=0)
        n_frames = T
    return torch.from_numpy(clipped), n_frames


def _infer_category(video_id: str) -> str:
    vid = video_id.replace("_x264", "")
    match = re.match(r'^([A-Za-z_]+?)(\d)', vid)
    if match:
        return match.group(1).rstrip("_")
    return "Unknown"


# ---------------------------------------------------------------------------
# MIL collate: pair one anomaly clip with one normal clip per batch element
# ---------------------------------------------------------------------------


def mil_collate_fn(samples: list[FeatureSample]) -> dict[str, torch.Tensor]:
    """Stack samples into batched tensors.

    Returns a dict with keys:
        features   (B, T, yolo_dim+768)  — concatenated (backward compat)
        yolo       (B, T, yolo_dim)      — YOLO object features
        vit        (B, T, 768)           — VideoMAE clip embeddings
        flow       (B, T, 16)            — optical-flow features
        labels     (B, T)
        is_anomaly (B,) bool
        frame_indices (B, T) int32
        video_ids  list[str]
    """
    features      = torch.stack([s.features   for s in samples])
    yolo          = torch.stack([s.yolo       for s in samples])
    vit           = torch.stack([s.vit        for s in samples])
    flow          = torch.stack([s.flow       for s in samples])
    labels        = torch.stack([s.labels     for s in samples])
    is_anomaly    = torch.tensor([s.is_anomaly for s in samples], dtype=torch.bool)
    frame_indices = np.stack([s.frame_indices for s in samples])  # (B, T) int32
    video_ids     = [s.video_id for s in samples]
    return {
        "features": features, "yolo": yolo, "vit": vit, "flow": flow,
        "labels": labels, "is_anomaly": is_anomaly,
        "frame_indices": frame_indices, "video_ids": video_ids,
    }
