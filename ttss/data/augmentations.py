"""Temporal Threat Scoring System (TTSS): video augmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(slots=True)
class VideoAugmentationPipeline:
    """Composable video augmentation pipeline for TTSS clips."""

    horizontal_flip_prob: float = 0.0
    temporal_stride: int = 1
    normalize_frames: bool = False

    def __call__(self, frames: Sequence[Any]) -> list[Any]:
        """Apply the configured augmentations to a sequence of frames."""
        processed = self.temporal_subsample(frames, step=self.temporal_stride)
        processed = self.random_horizontal_flip(processed)
        if self.normalize_frames:
            processed = self.normalize(processed)
        return processed

    def temporal_subsample(self, frames: Sequence[Any], step: int = 1) -> list[Any]:
        """Subsample frames along the temporal axis."""
        if step <= 0:
            raise ValueError("step must be a positive integer")
        return list(frames[::step])

    def random_horizontal_flip(self, frames: Sequence[Any]) -> list[Any]:
        """Return frames unchanged until the image backend is wired in."""
        return list(frames)

    def normalize(self, frames: Sequence[Any]) -> list[Any]:
        """Return normalized frames once the tensor backend is configured."""
        return list(frames)
