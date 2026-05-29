"""Temporal Threat Scoring System (TTSS): temporal evaluation routines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

UCF_CRIME_CATEGORIES: list[str] = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


@dataclass
class CategoryReport:
    """Per-crime-category evaluation metrics."""

    category: str
    auc: float
    ear: float
    malt_frames: float
    n_videos: int
    precrime_ap: float = 0.0


@dataclass(slots=True)
class EvaluationReport:
    """Aggregated evaluation metrics for a TTSS run."""

    frame_level_auc: float
    early_alert_rate: float
    mean_alert_lead_time: float
    precrime_ap: float


def evaluate_sequence(
    y_true_crime: np.ndarray,
    y_score: np.ndarray,
    y_true_precrime: np.ndarray | None = None,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Compute all four evaluation metrics for a single temporal sequence.

    Parameters
    ----------
    y_true_crime:   1 for crime frames, 0 otherwise.
    y_score:        Continuous threat scores in [0, 1].
    y_true_precrime: 1 for pre-crime frames; inferred from y_true_crime when None.
    threshold:      Alert decision threshold.
    """
    from ttss.training.metrics import (  # local import avoids circular dependency
        early_alert_rate,
        frame_level_auc,
        mean_alert_lead_time,
        precrime_ap,
    )

    y_true_crime = np.asarray(y_true_crime)
    y_score = np.asarray(y_score, dtype=float)

    if y_true_precrime is None:
        onset_indices = np.where(y_true_crime == 1)[0]
        y_true_precrime = np.zeros_like(y_true_crime)
        if len(onset_indices) > 0:
            onset = int(onset_indices[0])
            y_true_precrime[:onset] = 1

    return EvaluationReport(
        frame_level_auc=frame_level_auc(y_true_crime, y_score),
        early_alert_rate=early_alert_rate(y_true_crime, y_score, threshold),
        mean_alert_lead_time=mean_alert_lead_time(y_true_crime, y_score, threshold),
        precrime_ap=precrime_ap(np.asarray(y_true_precrime), y_score),
    )


@dataclass(slots=True)
class TemporalEvaluationResult:
    """Lead-time evaluation output for pre-crime detection."""

    first_alarm_index: int | None
    lead_time_frames: int
    detected_pre_crime: bool


class TemporalEvaluator:
    """Evaluate pre-crime detection behavior over temporal windows."""

    def evaluate(
        self,
        scores: Sequence[float],
        crime_start_index: int,
        threshold: float = 0.5,
        pre_window: int = 30,
    ) -> TemporalEvaluationResult:
        """Measure whether a pre-crime alarm is raised before the event start."""
        window_start = max(0, crime_start_index - pre_window)
        for index in range(window_start, min(crime_start_index, len(scores))):
            if scores[index] >= threshold:
                return TemporalEvaluationResult(
                    first_alarm_index=index,
                    lead_time_frames=crime_start_index - index,
                    detected_pre_crime=True,
                )
        return TemporalEvaluationResult(
            first_alarm_index=None,
            lead_time_frames=0,
            detected_pre_crime=False,
        )


def per_category_metrics(
    results: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    threshold: float = 0.5,
    fps: float = 30.0,
) -> list[CategoryReport]:
    """Compute AUC, EAR, MALT, and precrime-AP broken down by crime category.

    Parameters
    ----------
    results:
        Mapping from category name to a list of ``(y_true, y_score)`` pairs,
        one per video.  ``y_true`` is 1 for crime frames, 0 otherwise.
    threshold:
        Alert decision threshold for EAR and MALT.
    fps:
        Frame rate used to convert MALT from frames to seconds (informational
        only — ``malt_frames`` is always returned in frames).

    Returns
    -------
    List of :class:`CategoryReport`, one per category present in *results*,
    sorted by AUC descending.
    """
    from ttss.training.metrics import (
        frame_level_auc,
        early_alert_rate,
        mean_alert_lead_time,
        precrime_ap as _precrime_ap,
    )

    reports: list[CategoryReport] = []
    for category, video_pairs in results.items():
        if not video_pairs:
            continue

        aucs, ears, malts, aps = [], [], [], []
        for y_true, y_score in video_pairs:
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score, dtype=float)

            aucs.append(frame_level_auc(y_true, y_score))
            ears.append(early_alert_rate(y_true, y_score, threshold))
            malts.append(mean_alert_lead_time(y_true, y_score, threshold))

            # pre-crime AP: label pre-crime frames as positive
            onset_idx = np.where(y_true == 1)[0]
            if len(onset_idx) > 0:
                onset = int(onset_idx[0])
                y_pre = np.zeros_like(y_true)
                y_pre[:onset] = 1
                aps.append(_precrime_ap(y_pre, y_score))

        reports.append(CategoryReport(
            category=category,
            auc=float(np.mean(aucs)),
            ear=float(np.mean(ears)),
            malt_frames=float(np.mean(malts)),
            n_videos=len(video_pairs),
            precrime_ap=float(np.mean(aps)) if aps else 0.0,
        ))

    reports.sort(key=lambda r: r.auc, reverse=True)
    return reports
