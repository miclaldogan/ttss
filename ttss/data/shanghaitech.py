"""Temporal Threat Scoring System (TTSS): ShanghaiTech-Campus dataset interface.

ShanghaiTech-Campus is a campus surveillance anomaly detection dataset with
13 scenes and 437 videos (130 anomaly, 307 normal).  Ground-truth is provided
as per-frame binary labels in .npy files.

Expected directory layout::

    <data_root>/
        frames/
            <scene_id>_<clip_id>/
                0001.jpg  0002.jpg  ...
        test_frame_mask/
            <scene_id>_<clip_id>.npy   # 1 = anomaly, 0 = normal
        test_meta.json                 # optional video metadata

Reference: Liu et al., "Future Frame Prediction for Anomaly Detection", CVPR 2018.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(slots=True)
class SHTRecord:
    """Metadata for a single ShanghaiTech clip."""

    clip_id: str
    scene_id: str
    split: str
    frames_dir: str
    label_path: str
    total_frames: int | None = None
    is_anomaly: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SHTSample:
    """Materialized sample from the ShanghaiTech dataset."""

    record: SHTRecord
    frame_indices: list[int]
    frame_labels: list[int]
    frames: list[Any]


class ShanghaiTechDataset:
    """Zero-shot evaluation dataset for ShanghaiTech-Campus.

    This loader follows the same interface as :class:`UcfCrimeDataset` so that
    any TTSS model can be evaluated on ShanghaiTech without changes.

    Parameters
    ----------
    records:       Pre-built list of :class:`SHTRecord`.
    data_root:     Root directory of the dataset.
    transform:     Optional frame transform (same signature as UCF-Crime).
    frame_stride:  Sample every *n*-th frame.
    max_frames:    Cap on frames per clip.
    load_frames:   When False skip actual frame loading (label-only mode).
    """

    DATASET_NAME = "ShanghaiTech-Campus"

    def __init__(
        self,
        records: Sequence[SHTRecord],
        data_root: str | Path,
        transform: Callable | None = None,
        frame_stride: int = 1,
        max_frames: int | None = None,
        load_frames: bool = True,
    ) -> None:
        self.records = list(records)
        self.data_root = Path(data_root)
        self.transform = transform
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.load_frames = load_frames

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        data_root: str | Path,
        split: str = "test",
        frame_stride: int = 1,
        max_frames: int | None = None,
        load_frames: bool = True,
    ) -> "ShanghaiTechDataset":
        """Discover clips by scanning the standard ShanghaiTech directory layout."""
        root = Path(data_root)
        mask_dir = root / "test_frame_mask"
        frames_base = root / "frames"
        records: list[SHTRecord] = []

        if not mask_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {mask_dir}")

        for label_path in sorted(mask_dir.glob("*.npy")):
            clip_id = label_path.stem
            scene_id = clip_id.split("_")[0] if "_" in clip_id else clip_id
            frames_dir = frames_base / clip_id

            labels = np.load(label_path)
            is_anomaly = bool(labels.max() > 0)

            records.append(SHTRecord(
                clip_id=clip_id,
                scene_id=scene_id,
                split=split,
                frames_dir=str(frames_dir),
                label_path=str(label_path),
                total_frames=len(labels),
                is_anomaly=is_anomaly,
            ))

        return cls(records, data_root, frame_stride=frame_stride,
                   max_frames=max_frames, load_frames=load_frames)

    @classmethod
    def from_meta_json(
        cls,
        meta_path: str | Path,
        data_root: str | Path,
        **kwargs,
    ) -> "ShanghaiTechDataset":
        """Build dataset from a metadata JSON file."""
        with open(meta_path) as f:
            items = json.load(f)
        records = [
            SHTRecord(
                clip_id=str(item["clip_id"]),
                scene_id=str(item.get("scene_id", item["clip_id"].split("_")[0])),
                split=str(item.get("split", "test")),
                frames_dir=str(item.get("frames_dir", "")),
                label_path=str(item.get("label_path", "")),
                total_frames=item.get("total_frames"),
                is_anomaly=bool(item.get("is_anomaly", False)),
            )
            for item in items
        ]
        return cls(records, data_root, **kwargs)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SHTSample:
        record = self.records[index]
        labels_arr = np.load(record.label_path)
        total = len(labels_arr)
        indices = list(range(0, total, self.frame_stride))
        if self.max_frames:
            indices = indices[: self.max_frames]

        frame_labels = [int(labels_arr[i]) for i in indices]
        frames: list[Any] = []

        if self.load_frames:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("opencv-python is required for frame loading") from exc
            frames_path = Path(record.frames_dir)
            img_files = sorted(frames_path.glob("*.jpg")) + sorted(frames_path.glob("*.png"))
            for i in indices:
                if i < len(img_files):
                    frames.append(cv2.imread(str(img_files[i])))

        if self.transform and frames:
            frames = list(self.transform(frames))

        return SHTSample(
            record=record,
            frame_indices=indices,
            frame_labels=frame_labels,
            frames=frames,
        )

    def clip_ids(self) -> list[str]:
        return [r.clip_id for r in self.records]

    def anomaly_records(self) -> list[SHTRecord]:
        return [r for r in self.records if r.is_anomaly]

    def normal_records(self) -> list[SHTRecord]:
        return [r for r in self.records if not r.is_anomaly]
