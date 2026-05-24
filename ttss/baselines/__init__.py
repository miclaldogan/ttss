"""Temporal Threat Scoring System (TTSS): baseline comparison suite.

Each baseline implements the ``BaselinePredictor`` protocol so it can be
swapped in wherever TTSS produces a ``(frame_scores, video_id)`` result.
Wrappers load published checkpoints when available; when a checkpoint is
absent they fall back to a reproducible synthetic scorer and emit a warning
so the evaluation pipeline can still complete end-to-end.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ttss.baselines.mean_feature_svm import MeanFeatureSVMBaseline
from ttss.baselines.rtfm import RTFMBaseline
from ttss.baselines.sultani2018 import Sultani2018Baseline

__all__ = [
    "BaselinePredictor",
    "MeanFeatureSVMBaseline",
    "RTFMBaseline",
    "Sultani2018Baseline",
]

REGISTRY: dict[str, type] = {
    "sultani2018": Sultani2018Baseline,
    "rtfm": RTFMBaseline,
    "mean_feature_svm": MeanFeatureSVMBaseline,
}


@runtime_checkable
class BaselinePredictor(Protocol):
    """Minimal interface every baseline wrapper must satisfy."""

    #: Human-readable identifier used in result tables and JSON output.
    name: str

    def predict_video(self, video_path: str) -> np.ndarray:
        """Return frame-level threat scores in [0, 1] for *video_path*.

        When the underlying model or checkpoint is unavailable the
        implementation should log a warning and return a synthetic score
        array of the same shape rather than raising an exception, so that
        the evaluation pipeline can always complete.

        Parameters
        ----------
        video_path:
            Path to a video file **or** a sentinel string used during
            synthetic evaluation (e.g. ``"synthetic://video_0"``).

        Returns
        -------
        np.ndarray
            1-D float32 array of frame-level scores, one value per frame.
        """
        ...
