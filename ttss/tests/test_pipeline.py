"""Temporal Threat Scoring System (TTSS): tests for the model pipeline."""

from typing import Any, Sequence

import torch

from ttss.models.detection.vit_scene import SceneEmbedding
from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor, ThreatPrediction
from ttss.models.recognition.yolov8_wrapper import Detection, DetectionBox
from ttss.models.ttss_pipeline import TtssPipeline, TTPipeline, _consecutive_warning_flags
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


def test_ucf_crime_dataset_sample_structure() -> None:
    from ttss.data.temporal_labeler import TemporalSpan
    from ttss.data.ucf_crime import AnnotationRecord, UcfCrimeDataset

    annotations = [
        AnnotationRecord(
            video_id="test_video_001",
            label="Robbery",
            fps=30.0,
            split="train",
            video_path="test_video_001.mp4",
            anomaly_spans=[TemporalSpan(start_frame=50, end_frame=100)],
            total_frames=200,
        )
    ]
    dataset = UcfCrimeDataset(
        annotations=annotations,
        data_root="/tmp",
        load_frames=False,
    )

    assert len(dataset) == 1

    sample = dataset[0]

    # Identity checks
    assert sample.annotation.video_id == "test_video_001"
    assert sample.annotation.label == "Robbery"

    # Shape: all per-frame lists must have length == total_frames
    assert len(sample.frame_indices) == 200
    assert len(sample.temporal_labels) == 200
    assert len(sample.threat_scores) == 200

    # All scores must be in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in sample.threat_scores)

    # The three threat zones must be present
    assert "pre_crime" in sample.temporal_labels
    assert "crime" in sample.temporal_labels
    assert "post_crime" in sample.temporal_labels

    # Frames list is empty when load_frames=False
    assert sample.frames == []


# ---------------------------------------------------------------------------
# TTPipeline alias
# ---------------------------------------------------------------------------


def test_ttpipeline_alias_is_same_class() -> None:
    assert TTPipeline is TtssPipeline


def test_ttpipeline_has_predict_method() -> None:
    assert callable(getattr(TTPipeline, "predict", None))


# ---------------------------------------------------------------------------
# early_warning_flags — consecutive logic
# ---------------------------------------------------------------------------


def test_consecutive_flags_three_in_a_row() -> None:
    """3 consecutive frames above threshold must all be flagged."""
    scores = [0.2, 0.7, 0.8, 0.9, 0.3]
    flags = _consecutive_warning_flags(scores, threshold=0.6, min_run=3)
    assert flags == [False, True, True, True, False]


def test_consecutive_flags_two_not_enough() -> None:
    """A run of only 2 consecutive frames must NOT be flagged."""
    scores = [0.7, 0.8, 0.3, 0.2]
    flags = _consecutive_warning_flags(scores, threshold=0.6, min_run=3)
    assert flags == [False, False, False, False]


def test_consecutive_flags_exactly_three() -> None:
    scores = [0.7, 0.8, 0.9]
    flags = _consecutive_warning_flags(scores, threshold=0.6, min_run=3)
    assert flags == [True, True, True]


def test_consecutive_flags_disjoint_runs() -> None:
    """Two separate short runs must not be merged."""
    scores = [0.7, 0.8, 0.2, 0.7, 0.8]
    flags = _consecutive_warning_flags(scores, threshold=0.6, min_run=3)
    assert flags == [False, False, False, False, False]


def test_consecutive_flags_long_run() -> None:
    scores = [0.1, 0.8, 0.9, 0.7, 0.8, 0.9, 0.1]
    flags = _consecutive_warning_flags(scores, threshold=0.6, min_run=3)
    assert flags == [False, True, True, True, True, True, False]


def test_consecutive_flags_all_below() -> None:
    scores = [0.1, 0.2, 0.3]
    flags = _consecutive_warning_flags(scores, threshold=0.6, min_run=3)
    assert flags == [False, False, False]


def test_consecutive_flags_empty() -> None:
    assert _consecutive_warning_flags([], 0.6) == []


# ---------------------------------------------------------------------------
# early_warning_flags populated on ThreatTimeline
# ---------------------------------------------------------------------------


class DummyPredictionModelHigh:
    """Returns 5 consecutive high scores for early-warning tests."""

    def predict_sequence(
        self,
        sequence_features: "Sequence[Sequence[float]] | torch.Tensor",
    ) -> ThreatPrediction:
        T = 5
        scores = torch.tensor([[0.8] * T], dtype=torch.float32)
        attn = torch.ones(1, T) / T
        return ThreatPrediction(
            frame_scores=scores,
            sequence_score=torch.tensor([0.8]),
            attention_weights=attn,
            hidden_state=torch.empty(0),
        )


def test_threat_timeline_has_early_warning_flags_field() -> None:
    pipeline = TtssPipeline(
        recognition_model=DummyRecognitionModel(),
        detection_model=DummyDetectionModel(),
        prediction_model=DummyPredictionModelHigh(),
        early_warning_threshold=0.6,
    )
    timeline = pipeline.predict_from_frames([object()] * 5)
    assert hasattr(timeline, "early_warning_flags")
    assert len(timeline.early_warning_flags) == 5


def test_early_warning_flags_all_true_for_high_scores() -> None:
    pipeline = TtssPipeline(
        recognition_model=DummyRecognitionModel(),
        detection_model=DummyDetectionModel(),
        prediction_model=DummyPredictionModelHigh(),
        early_warning_threshold=0.6,
    )
    timeline = pipeline.predict_from_frames([object()] * 5)
    assert all(timeline.early_warning_flags)


def test_early_warning_flags_false_for_two_frame_run() -> None:
    """Pipeline with only 2 high-score frames must not raise early warning flags."""

    class Two:
        def predict_sequence(self, seq):
            scores = torch.tensor([[0.8, 0.9]], dtype=torch.float32)
            attn = torch.tensor([[0.5, 0.5]])
            return ThreatPrediction(
                frame_scores=scores,
                sequence_score=torch.tensor([0.85]),
                attention_weights=attn,
                hidden_state=torch.empty(0),
            )

    pipeline = TtssPipeline(
        recognition_model=DummyRecognitionModel(),
        detection_model=DummyDetectionModel(),
        prediction_model=Two(),
        early_warning_threshold=0.6,
    )
    timeline = pipeline.predict_from_frames([object()] * 2)
    assert timeline.early_warning_flags == [False, False]


# ---------------------------------------------------------------------------
# demo.py arg parsing
# ---------------------------------------------------------------------------


def test_demo_parser_accepts_video_flag() -> None:
    from ttss.scripts.demo import build_parser
    args = build_parser().parse_args(["--video", "sample.mp4"])
    assert args.video == "sample.mp4"


def test_demo_parser_default_threshold() -> None:
    from ttss.scripts.demo import build_parser
    args = build_parser().parse_args(["--video", "x.mp4"])
    assert args.threshold == 0.6
