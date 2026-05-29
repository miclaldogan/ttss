"""Temporal Threat Scoring System (TTSS): UCF-Crime dataset interface.

This module loads raw annotation metadata, extracts video frames when requested,
and materializes per-frame temporal tags and threat scores for TTSS training.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import cv2
except ImportError:  # pragma: no cover - optional during lightweight unit tests
    cv2 = None

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - fallback for environments without torch
    class Dataset:  # type: ignore[no-redef]
        """Fallback dataset base when torch is unavailable."""

        pass

from ttss.data.temporal_labeler import (
    NORMAL_LABEL,
    TemporalSpan,
    TemporalThreatLabeler,
)


FrameTransform = Callable[[Sequence[Any]], Sequence[Any]]


@dataclass(slots=True)
class AnnotationRecord:
    """Metadata for a single annotated UCF-Crime video."""

    video_id: str
    label: str
    fps: float
    split: str
    video_path: str
    anomaly_spans: list[TemporalSpan] = field(default_factory=list)
    total_frames: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def crime_start_frame(self) -> int:
        """Return the first anomaly start frame for compatibility."""

        if not self.anomaly_spans:
            return 0
        return self.anomaly_spans[0].start_frame

    @property
    def crime_end_frame(self) -> int:
        """Return the last anomaly end frame for compatibility."""

        if not self.anomaly_spans:
            return 0
        return self.anomaly_spans[-1].end_frame


@dataclass(slots=True)
class UcfCrimeSample:
    """Materialized sample returned by the dataset."""

    annotation: AnnotationRecord
    frame_indices: list[int]
    temporal_labels: list[str]
    threat_scores: list[float]
    frames: list[Any]


class UcfCrimeDataset(Dataset):
    """Dataset wrapper for UCF-Crime metadata and video samples."""

    def __init__(
        self,
        annotations: Sequence[AnnotationRecord],
        data_root: str | Path,
        transform: FrameTransform | None = None,
        labeler: TemporalThreatLabeler | None = None,
        frame_stride: int = 1,
        max_frames: int | None = None,
        load_frames: bool = True,
        split_file: str | Path | None = None,
    ) -> None:
        if frame_stride <= 0:
            raise ValueError("frame_stride must be a positive integer")

        all_annotations = list(annotations)

        # Filter to split_file video IDs when provided
        if split_file is not None:
            split_path = Path(split_file)
            if not split_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_path}")
            split_ids = {
                line.strip().split("/")[-1].replace("_x264.mp4", "").replace(".mp4", "")
                for line in split_path.read_text().splitlines()
                if line.strip()
            }
            all_annotations = [
                a for a in all_annotations
                if a.video_id in split_ids or a.video_path.split("/")[-1].replace(".mp4", "") in split_ids
            ]

        self.annotations = all_annotations
        self.data_root = Path(data_root)
        self.transform = transform
        self.labeler = labeler or TemporalThreatLabeler()
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.load_frames = load_frames
        self.split_file = split_file

    @classmethod
    def from_annotation_csv(
        cls,
        annotation_file: str | Path,
        data_root: str | Path,
        transform: FrameTransform | None = None,
        labeler: TemporalThreatLabeler | None = None,
        frame_stride: int = 1,
        max_frames: int | None = None,
        load_frames: bool = True,
    ) -> "UcfCrimeDataset":
        """Build the dataset from a CSV metadata file."""
        rows: list[AnnotationRecord] = []
        with Path(annotation_file).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                anomaly_spans = [
                    TemporalSpan(
                        start_frame=int(row.get("crime_start_frame", 0)),
                        end_frame=int(row.get("crime_end_frame", 0)),
                    )
                ]
                rows.append(
                    AnnotationRecord(
                        video_id=row["video_id"],
                        label=row["label"],
                        anomaly_spans=anomaly_spans,
                        fps=float(row.get("fps", 30.0)),
                        split=row.get("split", "train"),
                        video_path=row.get("video_path", f"{row['video_id']}.mp4"),
                        total_frames=_optional_int(row.get("total_frames")),
                    )
                )
        return cls(
            rows,
            data_root,
            transform,
            labeler,
            frame_stride,
            max_frames,
            load_frames,
        )

    @classmethod
    def from_annotation_json(
        cls,
        annotation_file: str | Path,
        data_root: str | Path,
        transform: FrameTransform | None = None,
        labeler: TemporalThreatLabeler | None = None,
        frame_stride: int = 1,
        max_frames: int | None = None,
        load_frames: bool = True,
    ) -> "UcfCrimeDataset":
        """Build the dataset from a prepared TTSS annotation JSON file."""
        records = load_annotation_records(annotation_file)
        return cls(
            records,
            data_root,
            transform,
            labeler,
            frame_stride,
            max_frames,
            load_frames,
        )

    def __len__(self) -> int:
        """Return the number of annotated videos."""

        return len(self.annotations)

    def __getitem__(self, index: int) -> UcfCrimeSample:
        """Return a typed dataset sample for the given index."""
        annotation = self.annotations[index]
        video_path = self.resolve_video_path(annotation.video_path)

        frames: list[Any] = []
        frame_indices: list[int] = []
        if self.load_frames:
            frame_indices, frames, detected_total = self.extract_frames(video_path)
            if annotation.total_frames is None:
                annotation.total_frames = detected_total
        else:
            total_frames = annotation.total_frames or self.inspect_video(video_path)["total_frames"]
            annotation.total_frames = total_frames
            frame_indices = self.sample_frame_indices(total_frames)

        if self.transform is not None:
            frames = list(self.transform(frames))

        labels = self.labeler.label_intervals(
            total_frames=annotation.total_frames or len(frame_indices) or 1,
            anomaly_spans=annotation.anomaly_spans,
        )
        return UcfCrimeSample(
            annotation=annotation,
            frame_indices=frame_indices,
            temporal_labels=[labels[frame_index].label for frame_index in frame_indices],
            threat_scores=[labels[frame_index].threat_score for frame_index in frame_indices],
            frames=frames,
        )

    def video_ids(self) -> list[str]:
        """Return the ordered list of video identifiers."""

        return [annotation.video_id for annotation in self.annotations]

    def resolve_video_path(self, video_path: str) -> Path:
        """Resolve relative paths against the configured dataset root."""
        candidate = Path(video_path)
        if candidate.is_absolute():
            return candidate
        return self.data_root / candidate

    def inspect_video(self, video_path: str | Path) -> dict[str, float | int]:
        """Read basic video metadata using OpenCV when available."""
        if cv2 is None:
            raise RuntimeError("opencv-python is required for video inspection")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.release()
        return {"total_frames": total_frames, "fps": fps}

    def sample_frame_indices(self, total_frames: int) -> list[int]:
        """Sample frame indices according to dataset stride and cap settings."""
        indices = list(range(0, total_frames, self.frame_stride))
        if self.max_frames is not None:
            return indices[: self.max_frames]
        return indices

    def extract_frames(self, video_path: str | Path) -> tuple[list[int], list[Any], int]:
        """Extract frames from a video file using OpenCV."""
        if cv2 is None:
            raise RuntimeError("opencv-python is required for frame extraction")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_indices = self.sample_frame_indices(total_frames)
        selected = set(sampled_indices)
        frames: list[Any] = []
        frame_indices: list[int] = []
        current_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if current_index in selected:
                frame_indices.append(current_index)
                frames.append(frame)
                if self.max_frames is not None and len(frame_indices) >= self.max_frames:
                    break
            current_index += 1

        capture.release()
        return frame_indices, frames, total_frames


def load_annotation_records(annotation_file: str | Path) -> list[AnnotationRecord]:
    """Load TTSS annotation records from JSON or CSV metadata files."""
    path = Path(annotation_file)
    if path.suffix.lower() == ".csv":
        return UcfCrimeDataset.from_annotation_csv(
            annotation_file=path,
            data_root=path.parent,
            load_frames=False,
        ).annotations

    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    items: list[dict[str, Any]]
    if isinstance(document, list):
        items = document
    elif isinstance(document, dict) and isinstance(document.get("videos"), list):
        items = document["videos"]
    else:
        items = [document]

    return [_record_from_json(item) for item in items]


def _record_from_json(item: dict[str, Any]) -> AnnotationRecord:
    spans = item.get("anomaly_spans") or _spans_from_segments(item.get("segments", {}))
    anomaly_spans = [
        TemporalSpan(
            start_frame=int(span["start_frame"]),
            end_frame=int(span["end_frame"]),
        )
        for span in spans
    ]
    return AnnotationRecord(
        video_id=str(item["video_id"]),
        label=str(item.get("label", NORMAL_LABEL)),
        anomaly_spans=anomaly_spans,
        fps=float(item.get("fps", 30.0)),
        split=str(item.get("split", "train")),
        video_path=str(item.get("video_path", f"{item['video_id']}.mp4")),
        total_frames=_optional_int(item.get("total_frames")),
        metadata=dict(item.get("metadata", {})),
    )


def _spans_from_segments(segments: dict[str, Any]) -> list[dict[str, int]]:
    crime_segment = segments.get("crime")
    if crime_segment is None:
        return []
    if isinstance(crime_segment, list):
        return crime_segment
    return [crime_segment]


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
