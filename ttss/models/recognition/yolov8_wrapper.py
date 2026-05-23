"""Temporal Threat Scoring System (TTSS): YOLOv8 recognition wrapper.

This module wraps YOLOv8 for frame-level object recognition and converts raw
detections into TTSS-friendly focused categories and numeric features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None


@dataclass(slots=True)
class Detection:
    """Detection output from the recognition layer."""

    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    frame_id: int = -1
    class_id: int = -1

    @property
    def area(self) -> float:
        """Return the area of the detection bounding box."""

        left, top, right, bottom = self.xyxy
        return max(0.0, right - left) * max(0.0, bottom - top)


DetectionBox = Detection


class YoloV8Wrapper:
    """YOLOv8 wrapper focused on person, vehicle, and weapon cues."""

    FEATURE_ORDER = (
        "person_count",
        "car_count",
        "weapon_count",
        "mean_confidence",
        "max_confidence",
        "mean_area",
        "person_confidence",
        "weapon_confidence",
    )

    CLASS_MAP = {
        "person": "person",
        "car": "car",
        "bus": "car",
        "truck": "car",
        "motorcycle": "car",
        "bicycle": "car",
        "knife": "weapon",
        "scissors": "weapon",
        "baseball bat": "weapon",
        "sports ball": "weapon",
    }

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        model_variant: str = "yolov8n",
        focus_classes: Sequence[str] | None = None,
        max_detections: int = 20,
        model: Any | None = None,
    ) -> None:
        self.weights_path = weights_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model_variant = model_variant
        self.focus_classes = tuple(focus_classes or ("person", "car", "weapon"))
        self.max_detections = max_detections
        self.model = model

    def load(self) -> None:
        """Load the YOLOv8 backend on demand."""
        if self.model is not None:
            return
        if YOLO is None:
            raise RuntimeError("ultralytics is required to load YOLOv8 weights")
        self.model = YOLO(self.weights_path)
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    @property
    def feature_dim(self) -> int:
        """Return the fixed recognition feature dimension."""

        return len(self.FEATURE_ORDER)

    def predict(self, frame: Any, frame_id: int = 0) -> list[Detection]:
        """Run focused detection on a single frame."""
        if self.model is None:
            self.load()

        raw_results = self.model.predict(
            source=frame,
            verbose=False,
            conf=self.confidence_threshold,
            device=self.device,
            max_det=self.max_detections,
        )
        detections: list[Detection] = []
        for result in raw_results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            cls_values = boxes.cls.detach().cpu().tolist()
            conf_values = boxes.conf.detach().cpu().tolist()
            xyxy_values = boxes.xyxy.detach().cpu().tolist()
            for class_id, confidence, xyxy in zip(
                cls_values,
                conf_values,
                xyxy_values,
                strict=True,
            ):
                raw_label = str(names.get(int(class_id), class_id))
                mapped_label = self.normalize_label(raw_label)
                if mapped_label not in self.focus_classes:
                    continue
                detections.append(
                    Detection(
                        label=mapped_label,
                        confidence=float(confidence),
                        xyxy=tuple(float(value) for value in xyxy),
                        frame_id=frame_id,
                        class_id=int(class_id),
                    )
                )
        return detections

    def predict_frames(self, frames: Sequence[Any]) -> dict[int, list[Detection]]:
        """Run focused detection on a sequence of frames."""

        return {
            frame_id: self.predict(frame, frame_id=frame_id)
            for frame_id, frame in enumerate(frames)
        }

    def batch_predict(self, frames: Sequence[Any]) -> list[list[Detection]]:
        """Run detection on a batch of frames for compatibility."""

        return [self.predict(frame, frame_id=frame_id) for frame_id, frame in enumerate(frames)]

    def extract_summary_features(
        self,
        detections: Sequence[Detection],
    ) -> list[float]:
        """Convert detections into a compact numeric feature vector."""
        features = self.extract_feature_tensor(detections)
        return features.detach().cpu().tolist()

    def extract_feature_tensor(
        self,
        detections: Sequence[Detection],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert detections into a fixed-size PyTorch feature tensor.

        Returns a 1-D Tensor of shape ``(8,)`` with dtype ``float32``.  The
        eight elements follow the order defined in ``FEATURE_ORDER``:

        0. ``person_count``    — number of person detections
        1. ``car_count``       — number of vehicle detections
        2. ``weapon_count``    — number of weapon detections
        3. ``mean_confidence`` — mean confidence across all detections
        4. ``max_confidence``  — highest confidence across all detections
        5. ``mean_area``       — mean bounding-box area across all detections
        6. ``person_confidence`` — sum of person detection confidences
        7. ``weapon_confidence`` — sum of weapon detection confidences
        """
        person_detections = [det for det in detections if det.label == "person"]
        car_detections = [det for det in detections if det.label == "car"]
        weapon_detections = [det for det in detections if det.label == "weapon"]

        confidence_values = [det.confidence for det in detections]
        area_values = [det.area for det in detections]
        feature_values = [
            float(len(person_detections)),
            float(len(car_detections)),
            float(len(weapon_detections)),
            self._safe_mean(confidence_values),
            max(confidence_values, default=0.0),
            self._safe_mean(area_values),
            sum(det.confidence for det in person_detections),
            sum(det.confidence for det in weapon_detections),
        ]
        return torch.tensor(feature_values, dtype=dtype)

    def normalize_label(self, raw_label: str) -> str:
        """Map detector labels into TTSS focus categories."""
        normalized = raw_label.strip().lower()
        if normalized in self.CLASS_MAP:
            return self.CLASS_MAP[normalized]
        if "gun" in normalized or "weapon" in normalized or "rifle" in normalized:
            return "weapon"
        return normalized

    def _safe_mean(self, values: Sequence[float]) -> float:
        return 0.0 if not values else float(sum(values) / len(values))
