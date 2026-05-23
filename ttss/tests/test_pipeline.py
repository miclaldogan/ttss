"""Temporal Threat Scoring System (TTSS): tests for the model pipeline."""

from typing import Any, Sequence

import torch

from ttss.models.detection.vit_scene import SceneEmbedding
from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor, ThreatPrediction
from ttss.models.recognition.yolov8_wrapper import Detection, DetectionBox
from ttss.models.ttss_pipeline import TtssPipeline
from ttss.training.losses import (
    PreCrimeDetectionLoss,
    TemporalConsistencyLoss,
    ThreatScoreRegressionLoss,
)


class DummyRecognitionModel:
    """Predictable recognition stub for pipeline tests."""

    feature_dim = 8

    def predict(self, frame: Any, frame_id: int = 0) -> list[DetectionBox]:
        del frame
        return [
            Detection(
                label="person",
                confidence=0.9,
                xyxy=(0.0, 0.0, 1.0, 1.0),
                frame_id=frame_id,
            )
        ]

    def predict_frames(self, frames: Sequence[Any]) -> dict[int, list[DetectionBox]]:
        return {
            frame_id: self.predict(frame, frame_id=frame_id)
            for frame_id, frame in enumerate(frames)
        }

    def extract_summary_features(
        self,
        detections: Sequence[DetectionBox],
    ) -> list[float]:
        return self.extract_feature_tensor(detections).tolist()

    def extract_feature_tensor(self, detections: Sequence[DetectionBox]) -> torch.Tensor:
        person_count = float(len(detections))
        return torch.tensor(
            [person_count, 0.0, 0.0, 0.75, 0.9, 1.0, 0.75, 0.0],
            dtype=torch.float32,
        )


class DummyDetectionModel:
    """Predictable scene encoder stub for pipeline tests."""

    def encode_frame(self, frame: Any, frame_id: int = 0) -> SceneEmbedding:
        del frame
        return SceneEmbedding(
            vector=torch.tensor([0.25, 0.5], dtype=torch.float32),
            label="street",
            frame_id=frame_id,
        )

    def encode_batch(self, frames: Sequence[Any]) -> torch.Tensor:
        return torch.tensor([[0.25, 0.5] for _ in frames], dtype=torch.float32)


class DummyPredictionModel:
    """Predictable temporal head stub for pipeline tests."""

    def predict_sequence(
        self,
        sequence_features: Sequence[Sequence[float]] | torch.Tensor,
    ) -> ThreatPrediction:
        feature_tensor = torch.as_tensor(sequence_features, dtype=torch.float32)
        if feature_tensor.ndim == 2:
            feature_tensor = feature_tensor.unsqueeze(0)
        frame_scores = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
        attention_weights = torch.tensor([[0.4, 0.6]], dtype=torch.float32)
        return ThreatPrediction(
            frame_scores=frame_scores,
            sequence_score=torch.tensor([0.5], dtype=torch.float32),
            attention_weights=attention_weights,
            hidden_state=feature_tensor,
        )


def test_pipeline_predicts_from_frames() -> None:
    pipeline = TtssPipeline(
        recognition_model=DummyRecognitionModel(),
        detection_model=DummyDetectionModel(),
        prediction_model=DummyPredictionModel(),
        early_warning_threshold=0.6,
    )

    timeline = pipeline.predict_from_frames([object(), object()])

    assert timeline.sequence_score == 0.5
    assert timeline.alarm_triggered is True
    assert timeline.alarm_frames == [1]
    assert len(timeline.threat_scores) == 2
    assert timeline.fused_features.shape == (2, 10)


def test_bilstm_predictor_returns_framewise_scores() -> None:
    predictor = BiLSTMThreatPredictor(input_dim=10, projection_dim=8, hidden_dim=4)
    features = torch.randn(2, 5, 10)

    prediction = predictor(features)

    assert prediction.frame_scores.shape == (2, 5)
    assert prediction.sequence_score.shape == (2,)
    assert prediction.attention_weights.shape == (2, 5)
    assert torch.all(prediction.frame_scores >= 0.0)
    assert torch.all(prediction.frame_scores <= 1.0)


def test_temporal_losses_return_positive_scalars() -> None:
    predictions = torch.tensor([[0.1, 0.4, 0.8]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    precrime_mask = torch.tensor([[False, True, False]])

    smoothness = TemporalConsistencyLoss()(predictions)
    regression = ThreatScoreRegressionLoss()(predictions, targets)
    precrime = PreCrimeDetectionLoss()(predictions, targets, precrime_mask)

    assert smoothness.item() >= 0.0
    assert regression.item() > 0.0
    assert precrime.item() > 0.0
