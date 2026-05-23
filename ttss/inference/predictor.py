"""Temporal Threat Scoring System (TTSS): inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ttss.models.ttss_pipeline import TtssPipeline


@dataclass(slots=True)
class InferenceRequest:
    """User-facing request for TTSS inference."""

    video_path: str
    frames: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class InferenceResult:
    """Structured inference output from the TTSS pipeline."""

    threat_score: float
    temporal_label: str


class ThreatPredictor:
    """Convenience wrapper for TTSS inference workflows."""

    def __init__(self, pipeline: TtssPipeline | None = None) -> None:
        self.pipeline = pipeline or TtssPipeline()

    def predict_frames(self, frames: Sequence[Any]) -> InferenceResult:
        """Predict a threat score for an in-memory frame sequence."""
        prediction = self.pipeline.predict_from_frames(frames)
        label = "elevated" if prediction.score >= 0.5 else "low-risk"
        return InferenceResult(threat_score=prediction.score, temporal_label=label)

    def predict_video(self, request: InferenceRequest) -> InferenceResult:
        """Predict a threat score for a video request skeleton."""
        return self.predict_frames(request.frames)
