"""Temporal Threat Scoring System (TTSS): XD-Violence dataset interface.

XD-Violence is a large-scale multi-scene violence detection dataset with
4,754 untrimmed videos across 6 violence categories plus normal videos.
Labels are provided as binary per-frame annotations.

Expected directory layout::

    <data_root>/
        videos/
            <video_id>.mp4  (or .avi)
        annotations/
            <video_id>.npy       # per-frame binary labels
        splits/
            train.txt            # one video_id per line
            test.txt

Violence categories: Fighting, Shooting, Riot, Abuse, Car accident, Explosion.

Reference: Wu et al., "Not Only Look, but Also Listen", ECCV 2020.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

XD_VIOLENCE_CATEGORIES: list[str] = [
    "Fighting", "Shooting", "Riot", "Abuse", "CarAccident", "Explosion",
]


@dataclass(slots=True)
class XDRecord:
    """Metadata for a single XD-Violence video."""

    video_id: str
    label: str
    split: str
    video_path: str
    label_path: str
    total_frames: int | None = None
    is_anomaly: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class XDSample:
    """Materialized sample from the XD-Violence dataset."""

    record: XDRecord
    frame_indices: list[int]
    frame_labels: list[int]
    frames: list[Any]


class XDViolenceDataset:
    """Zero-shot evaluation dataset for XD-Violence.

    Follows the same interface as :class:`UcfCrimeDataset` for drop-in
    cross-dataset evaluation.

    Parameters
    ----------
    records:      Pre-built list of :class:`XDRecord`.
    data_root:    Root directory of the dataset.
    transform:    Optional frame transform.
    frame_stride: Sample every *n*-th frame.
    max_frames:   Cap on frames per clip.
    load_frames:  When False skip frame loading (label-only mode).
    """

    DATASET_NAME = "XD-Violence"

    def __init__(
        self,
        records: Sequence[XDRecord],
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
    ) -> "XDViolenceDataset":
        """Discover videos from the standard XD-Violence layout."""
        root = Path(data_root)
        ann_dir = root / "annotations"
        video_dir = root / "videos"
        split_file = root / "splits" / f"{split}.txt"

        if not ann_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")

        video_ids: list[str] = []
        if split_file.exists():
            video_ids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        else:
            video_ids = [p.stem for p in sorted(ann_dir.glob("*.npy"))]

        records: list[XDRecord] = []
        video_extensions = [".mp4", ".avi", ".mkv", ".mov"]
        for vid_id in video_ids:
            label_path = ann_dir / f"{vid_id}.npy"
            if not label_path.exists():
                continue

            video_path = ""
            for ext in video_extensions:
                candidate = video_dir / f"{vid_id}{ext}"
                if candidate.exists():
                    video_path = str(candidate)
                    break

            labels = np.load(label_path)
            is_anomaly = bool(labels.max() > 0)
            category = _infer_category(vid_id)

            records.append(XDRecord(
                video_id=vid_id,
                label=category,
                split=split,
                video_path=video_path,
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
    ) -> "XDViolenceDataset":
        """Build dataset from a metadata JSON file."""
        with open(meta_path) as f:
            items = json.load(f)
        records = [
            XDRecord(
                video_id=str(item["video_id"]),
                label=str(item.get("label", "Unknown")),
                split=str(item.get("split", "test")),
                video_path=str(item.get("video_path", "")),
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

    def __getitem__(self, index: int) -> XDSample:
        record = self.records[index]
        labels_arr = np.load(record.label_path)
        total = len(labels_arr)
        indices = list(range(0, total, self.frame_stride))
        if self.max_frames:
            indices = indices[: self.max_frames]

        frame_labels = [int(labels_arr[i]) for i in indices]
        frames: list[Any] = []

        if self.load_frames and record.video_path:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("opencv-python is required for frame loading") from exc
            cap = cv2.VideoCapture(record.video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {record.video_path}")
            selected = set(indices)
            current = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if current in selected:
                    frames.append(frame)
                    if self.max_frames and len(frames) >= self.max_frames:
                        break
                current += 1
            cap.release()

        if self.transform and frames:
            frames = list(self.transform(frames))

        return XDSample(
            record=record,
            frame_indices=indices,
            frame_labels=frame_labels,
            frames=frames,
        )

    def video_ids(self) -> list[str]:
        return [r.video_id for r in self.records]

    def by_category(self) -> dict[str, list[XDRecord]]:
        """Group records by violence category."""
        groups: dict[str, list[XDRecord]] = {}
        for r in self.records:
            groups.setdefault(r.label, []).append(r)
        return groups


def _infer_category(video_id: str) -> str:
    """Infer XD-Violence category from video_id naming convention."""
    vid_lower = video_id.lower()
    for cat in XD_VIOLENCE_CATEGORIES:
        if cat.lower() in vid_lower:
            return cat
    return "Normal" if "normal" in vid_lower else "Unknown"
