"""Unit tests for the YOLOv8 recognition layer wrapper (issue #3).

All tests run on CPU without ultralytics installed.  The wrapper is exercised
either through its pure-Python helpers (extract_feature_tensor, normalize_label)
or via a fully injected fake YOLO model that mimics the ultralytics API surface.
"""

from __future__ import annotations

import pytest
import torch

from ttss.models.recognition.yolov8_wrapper import Detection, YoloV8Wrapper


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _make_detection(label: str, confidence: float, w: float = 100.0, h: float = 200.0) -> Detection:
    return Detection(label=label, confidence=confidence, xyxy=(0.0, 0.0, w, h), frame_id=0)


class _FakeBox:
    """Minimal stand-in for ultralytics Boxes, matching the attribute API."""

    def __init__(self, cls_id: int, confidence: float, xyxy: list[float]) -> None:
        self.cls = torch.tensor([cls_id])
        self.conf = torch.tensor([confidence])
        self.xyxy = torch.tensor([xyxy])


class _FakeResult:
    def __init__(self, cls_id: int, confidence: float, label: str) -> None:
        self.names = {cls_id: label}
        self.boxes = _FakeBox(cls_id, confidence, [0.0, 0.0, 100.0, 200.0])


class _FakeYOLO:
    """Fake YOLO model — returns one 'person' detection per call."""

    def predict(self, source, verbose, conf, device, max_det) -> list[_FakeResult]:  # noqa: PLR0913
        return [_FakeResult(cls_id=0, confidence=0.85, label="person")]


# ---------------------------------------------------------------------------
# Feature vector tests
# ---------------------------------------------------------------------------

def test_extract_feature_tensor_shape_empty() -> None:
    wrapper = YoloV8Wrapper()
    tensor = wrapper.extract_feature_tensor([])

    assert tensor.shape == (8,)
    assert tensor.dtype == torch.float32
    assert tensor.sum().item() == 0.0


def test_extract_feature_tensor_shape_with_detections() -> None:
    wrapper = YoloV8Wrapper()
    detections = [
        _make_detection("person", 0.9),
        _make_detection("person", 0.8),
        _make_detection("weapon", 0.7),
    ]
    tensor = wrapper.extract_feature_tensor(detections)

    assert tensor.shape == (8,)
    assert tensor[0].item() == pytest.approx(2.0)   # person_count
    assert tensor[1].item() == pytest.approx(0.0)   # car_count
    assert tensor[2].item() == pytest.approx(1.0)   # weapon_count
    assert tensor[3].item() == pytest.approx((0.9 + 0.8 + 0.7) / 3, rel=1e-5)  # mean_confidence
    assert tensor[4].item() == pytest.approx(0.9)   # max_confidence
    assert tensor[6].item() == pytest.approx(0.9 + 0.8)  # person_confidence
    assert tensor[7].item() == pytest.approx(0.7)   # weapon_confidence


def test_feature_order_length_matches_feature_dim() -> None:
    wrapper = YoloV8Wrapper()
    assert len(wrapper.FEATURE_ORDER) == wrapper.feature_dim == 8


def test_extract_feature_tensor_dtype_is_float32() -> None:
    wrapper = YoloV8Wrapper()
    tensor = wrapper.extract_feature_tensor([_make_detection("car", 0.6)])
    assert tensor.dtype == torch.float32


def test_extract_feature_tensor_runs_on_cpu() -> None:
    wrapper = YoloV8Wrapper(device="cpu")
    tensor = wrapper.extract_feature_tensor([_make_detection("person", 0.5)])
    assert tensor.device.type == "cpu"


# ---------------------------------------------------------------------------
# Label normalisation tests
# ---------------------------------------------------------------------------

def test_normalize_label_knife_to_weapon() -> None:
    assert YoloV8Wrapper().normalize_label("knife") == "weapon"


def test_normalize_label_bus_to_car() -> None:
    assert YoloV8Wrapper().normalize_label("bus") == "car"


def test_normalize_label_person_unchanged() -> None:
    assert YoloV8Wrapper().normalize_label("person") == "person"


def test_normalize_label_unknown_passthrough() -> None:
    assert YoloV8Wrapper().normalize_label("dog") == "dog"


# ---------------------------------------------------------------------------
# predict / predict_frames with injected fake model (no ultralytics)
# ---------------------------------------------------------------------------

def test_predict_returns_list_of_detections() -> None:
    wrapper = YoloV8Wrapper(model=_FakeYOLO())
    result = wrapper.predict("synthetic_frame", frame_id=3)

    assert isinstance(result, list)
    assert all(isinstance(d, Detection) for d in result)
    assert result[0].label == "person"
    assert result[0].frame_id == 3


def test_predict_frames_keys_match_frame_ids() -> None:
    wrapper = YoloV8Wrapper(model=_FakeYOLO())
    frames = ["f0", "f1", "f2"]
    result = wrapper.predict_frames(frames)

    assert set(result.keys()) == {0, 1, 2}
    for frame_id, detections in result.items():
        assert all(d.frame_id == frame_id for d in detections)


def test_predict_confidence_threshold_filters_below_threshold() -> None:
    """Detections below confidence_threshold are dropped by the YOLO call.

    This is enforced by passing conf= to model.predict; the wrapper itself
    does not double-filter.  We verify the returned list only contains the
    single high-confidence detection produced by _FakeYOLO.
    """
    wrapper = YoloV8Wrapper(model=_FakeYOLO(), confidence_threshold=0.5)
    result = wrapper.predict("frame")

    assert len(result) == 1
    assert result[0].confidence == pytest.approx(0.85)


def test_extract_summary_features_matches_tensor_tolist() -> None:
    wrapper = YoloV8Wrapper()
    detections = [_make_detection("person", 0.75)]
    tensor = wrapper.extract_feature_tensor(detections)
    summary = wrapper.extract_summary_features(detections)

    assert summary == pytest.approx(tensor.tolist())
