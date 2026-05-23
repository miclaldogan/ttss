"""Temporal Threat Scoring System (TTSS): end-to-end threat scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from ttss.models.detection.vit_scene import SceneEmbedding, VitSceneEncoder
from ttss.models.prediction.bilstm_threat import (
    BiLSTMThreatPredictor,
    ThreatPrediction,
)
from ttss.models.recognition.yolov8_wrapper import Detection, DetectionBox, YoloV8Wrapper


@dataclass(slots=True)
class ThreatTimeline:
    """Structured TTSS output for a video or frame sequence."""

    frame_ids: list[int]
    threat_scores: list[float]
    detections: dict[int, list[Detection]] = field(default_factory=dict)
    scene_embeddings: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    fused_features: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    attention_weights: list[float] = field(default_factory=list)
    sequence_score: float = 0.0
    alarm_triggered: bool = False
    alarm_frames: list[int] = field(default_factory=list)
    threshold: float = 0.6


class RecognitionModel(Protocol):
    """Protocol for TTSS recognition backends."""

    def predict(self, frame: Any, frame_id: int = 0) -> list[DetectionBox]:
        """Predict detections for a frame."""

    def predict_frames(self, frames: Sequence[Any]) -> dict[int, list[DetectionBox]]:
        """Predict detections for multiple frames."""

    def extract_summary_features(
        self,
        detections: Sequence[DetectionBox],
    ) -> list[float]:
        """Convert detections into fixed-size features."""

    def extract_feature_tensor(self, detections: Sequence[DetectionBox]) -> torch.Tensor:
        """Convert detections into a feature tensor."""


class DetectionModel(Protocol):
    """Protocol for TTSS scene encoders."""

    def encode_frame(self, frame: Any, frame_id: int = 0) -> SceneEmbedding:
        """Encode a frame into a scene embedding."""

    def encode_batch(self, frames: Sequence[Any]) -> torch.Tensor:
        """Encode a batch of frames."""


class PredictionModel(Protocol):
    """Protocol for TTSS temporal predictors."""

    def predict_sequence(
        self,
        sequence_features: Sequence[Sequence[float]] | torch.Tensor,
    ) -> ThreatPrediction:
        """Predict a sequence-level threat score."""


class TtssPipeline:
    """Connect recognition, detection, and prediction layers."""

    def __init__(
        self,
        recognition_model: RecognitionModel | None = None,
        detection_model: DetectionModel | None = None,
        prediction_model: PredictionModel | None = None,
        early_warning_threshold: float = 0.6,
    ) -> None:
        self.recognition_model = recognition_model or YoloV8Wrapper()
        self.detection_model = detection_model or VitSceneEncoder()
        self.prediction_model = prediction_model or BiLSTMThreatPredictor()
        self.early_warning_threshold = early_warning_threshold

    def load_video(
        self,
        video_path: str,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ) -> tuple[list[int], list[Any]]:
        """Load a video file and return sampled frame ids and frames."""
        if cv2 is None:
            raise RuntimeError("opencv-python is required for video loading")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be a positive integer")

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        frame_ids: list[int] = []
        frames: list[Any] = []
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_stride == 0:
                frame_ids.append(frame_index)
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_index += 1
        capture.release()
        return frame_ids, frames

    def build_fused_feature(
        self,
        recognition_features: Sequence[float] | torch.Tensor,
        scene_features: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        """Fuse recognition and scene features into a single vector."""
        recognition_tensor = torch.as_tensor(recognition_features, dtype=torch.float32)
        scene_tensor = torch.as_tensor(scene_features, dtype=torch.float32)
        return torch.cat([recognition_tensor, scene_tensor], dim=-1)

    def recognize_frames(self, frames: Sequence[Any]) -> dict[int, list[Detection]]:
        """Run the recognition layer over all frames."""
        if hasattr(self.recognition_model, "predict_frames"):
            return self.recognition_model.predict_frames(frames)
        return {
            frame_id: self.recognition_model.predict(frame, frame_id=frame_id)
            for frame_id, frame in enumerate(frames)
        }

    def detect_scenes(self, frames: Sequence[Any]) -> torch.Tensor:
        """Run the scene encoder over a batch of frames."""
        if hasattr(self.detection_model, "encode_batch"):
            embeddings = self.detection_model.encode_batch(frames)
            return torch.as_tensor(embeddings, dtype=torch.float32)
        encoded = [
            self.detection_model.encode_frame(frame, frame_id=frame_id).vector
            for frame_id, frame in enumerate(frames)
        ]
        return torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in encoded])

    def fuse_features(
        self,
        detections: dict[int, list[Detection]],
        scene_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse recognition features and scene embeddings for each frame."""
        fused_rows: list[torch.Tensor] = []
        for frame_id in range(scene_embeddings.shape[0]):
            detection_features = self.recognition_model.extract_feature_tensor(
                detections.get(frame_id, [])
            )
            fused_rows.append(
                self.build_fused_feature(
                    recognition_features=detection_features,
                    scene_features=scene_embeddings[frame_id],
                )
            )
        if not fused_rows:
            scene_dim = int(scene_embeddings.shape[-1]) if scene_embeddings.ndim == 2 else 0
            feature_dim = getattr(self.recognition_model, "feature_dim", 0)
            return torch.empty((0, feature_dim + scene_dim), dtype=torch.float32)
        return torch.stack(fused_rows, dim=0)

    def predict_from_frames(
        self,
        frames: Sequence[Any],
        frame_ids: Sequence[int] | None = None,
    ) -> ThreatTimeline:
        """Run the TTSS pipeline for a sequence of frames."""
        resolved_frame_ids = list(frame_ids) if frame_ids is not None else list(range(len(frames)))
        detections = self.recognize_frames(frames)
        scene_embeddings = self.detect_scenes(frames)
        fused_features = self.fuse_features(detections, scene_embeddings)
        prediction = self.prediction_model.predict_sequence(fused_features)
        frame_scores = prediction.frame_scores.detach().cpu().reshape(-1).tolist()
        alarm_frames = [
            frame_id
            for frame_id, score in zip(resolved_frame_ids, frame_scores, strict=True)
            if score > self.early_warning_threshold
        ]
        return ThreatTimeline(
            frame_ids=resolved_frame_ids,
            threat_scores=frame_scores,
            detections=detections,
            scene_embeddings=scene_embeddings.detach().cpu(),
            fused_features=fused_features.detach().cpu(),
            attention_weights=prediction.attention_weights.detach().cpu().reshape(-1).tolist(),
            sequence_score=prediction.score,
            alarm_triggered=bool(alarm_frames),
            alarm_frames=alarm_frames,
            threshold=self.early_warning_threshold,
        )

    def predict_video(
        self,
        video_path: str,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ) -> ThreatTimeline:
        """Load a video and run the full TTSS stack."""
        frame_ids, frames = self.load_video(
            video_path=video_path,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        return self.predict_from_frames(frames, frame_ids=frame_ids)

    def predict_from_features(
        self,
        sequence_features: Sequence[Sequence[float]] | torch.Tensor,
    ) -> ThreatPrediction:
        """Run the temporal threat model on precomputed features."""
        return self.prediction_model.predict_sequence(sequence_features)
