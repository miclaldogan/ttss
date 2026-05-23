"""Temporal Threat Scoring System (TTSS): data utility helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def load_annotation_rows(annotation_file: str | Path) -> list[dict[str, str]]:
    """Load raw annotation rows from a CSV file."""
    with Path(annotation_file).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_frame_range(start_frame: int, end_frame: int) -> None:
    """Validate that a frame interval is non-negative and ordered."""
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")


def build_sliding_windows(
    frame_indices: Iterable[int],
    window_size: int,
    stride: int,
) -> list[list[int]]:
    """Build index windows used for clip generation."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    indices = list(frame_indices)
    windows: list[list[int]] = []
    for start in range(0, max(len(indices) - window_size + 1, 0), stride):
        windows.append(indices[start : start + window_size])
    return windows
