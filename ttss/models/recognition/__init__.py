"""Temporal Threat Scoring System (TTSS): recognition model exports."""

from ttss.models.recognition.yolov8_wrapper import (
    Detection,
    DetectionBox,
    YoloV8Wrapper,
)

__all__ = ["Detection", "DetectionBox", "YoloV8Wrapper"]
