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

    features: torch.Tensor      # (T, 776) float32
    labels: torch.Tensor        # (T,) float32  — 0.0 or 1.0 (weak)
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

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> FeatureSample:
        path = self._files[idx]
        data = np.load(path, allow_pickle=True)

        yolo = data["yolo_features"].astype(np.float32)   # (T_raw, 8)
        vit  = data["vit_features"].astype(np.float32)    # (T_raw, 768)
        features_raw = np.concatenate([yolo, vit], axis=1)  # (T_raw, 776)

        video_id = str(data["video_id"])
        label    = str(data["label"])
        is_anomaly = label not in NORMAL_CATEGORIES

        T_raw = len(features_raw)
        features, n_frames = _clip_and_pad(features_raw, self.clip_length)

        # Weak label: 1.0 for all frames in anomaly video, 0.0 for normal
        # (weakly supervised — frame-level labels not used during training)
        label_val = 1.0 if is_anomaly else 0.0
        labels = torch.full((self.clip_length,), label_val, dtype=torch.float32)

        return FeatureSample(
            features=features,
            labels=labels,
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
    """Stack samples into batched tensors, keeping anomaly/normal separate.

    Returns a dict with keys:
        features  (B, T, 776)
        labels    (B, T)
        is_anomaly (B,) bool
    """
    features = torch.stack([s.features for s in samples])
    labels   = torch.stack([s.labels   for s in samples])
    is_anomaly = torch.tensor([s.is_anomaly for s in samples], dtype=torch.bool)
    return {"features": features, "labels": labels, "is_anomaly": is_anomaly}
