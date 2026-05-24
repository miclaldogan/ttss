"""Temporal Threat Scoring System (TTSS): threat score metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def binary_f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute F1 score for binary predictions without external dependencies."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0

    true_positive = sum(
        1
        for target, pred in zip(y_true, y_pred, strict=True)
        if target == 1 and pred == 1
    )
    false_positive = sum(
        1
        for target, pred in zip(y_true, y_pred, strict=True)
        if target == 0 and pred == 1
    )
    false_negative = sum(
        1
        for target, pred in zip(y_true, y_pred, strict=True)
        if target == 1 and pred == 0
    )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    if precision_denominator == 0 or recall_denominator == 0:
        return 0.0

    precision = true_positive / precision_denominator
    recall = true_positive / recall_denominator
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (
        precision + recall
    )


def roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute ROC AUC using pairwise ranking logic."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    positives = [score for target, score in zip(y_true, y_score, strict=True) if target]
    negatives = [score for target, score in zip(y_true, y_score, strict=True) if not target]
    if not positives or not negatives:
        return 0.0

    concordant = 0.0
    total_pairs = len(positives) * len(negatives)
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                concordant += 1.0
            elif positive_score == negative_score:
                concordant += 0.5
    return concordant / total_pairs


def early_alarm_rate(
    scores: Sequence[float],
    threshold: float = 0.5,
    pre_window: int | None = None,
) -> float:
    """Compute the fraction of early-window frames that exceed threshold."""
    if not scores:
        return 0.0
    window = list(scores if pre_window is None else scores[:pre_window])
    if not window:
        return 0.0
    alarms = sum(score >= threshold for score in window)
    return alarms / len(window)


def threshold_predictions(
    scores: Sequence[float],
    threshold: float = 0.5,
) -> list[int]:
    """Convert continuous threat scores to binary decisions."""
    return [1 if score >= threshold else 0 for score in scores]


# ---------------------------------------------------------------------------
# Numpy-signature evaluation metrics (issue #8)
# All accept (y_true: np.ndarray, y_score: np.ndarray) as primary signature.
# ---------------------------------------------------------------------------


def frame_level_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC: threat_score vs binary crime label.

    y_true: 1 for crime frames, 0 otherwise.
    Uses pairwise ranking — equivalent to sklearn's roc_auc_score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    concordant = float(np.sum(positives[:, None] > negatives[None, :])) + 0.5 * float(
        np.sum(positives[:, None] == negatives[None, :])
    )
    return concordant / (len(positives) * len(negatives))


def early_alert_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Early Alert Rate (EAR): fraction of pre-crime frames where score >= threshold.

    Pre-crime frames are those before the first crime onset derived from y_true.
    y_true: 1 for crime frames, 0 otherwise.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    onset_indices = np.where(y_true == 1)[0]
    if len(onset_indices) == 0 or onset_indices[0] == 0:
        return 0.0
    onset = int(onset_indices[0])
    return float(np.mean(y_score[:onset] >= threshold))


def mean_alert_lead_time(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Mean Alert Lead Time (MALT): frames between the first pre-crime alert and crime onset.

    y_true: 1 for crime frames, 0 otherwise.
    Returns 0.0 when no alert fires before onset.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    onset_indices = np.where(y_true == 1)[0]
    if len(onset_indices) == 0 or onset_indices[0] == 0:
        return 0.0
    onset = int(onset_indices[0])
    pre_alerts = np.where(y_score[:onset] >= threshold)[0]
    if len(pre_alerts) == 0:
        return 0.0
    return float(onset - int(pre_alerts[0]))


def precrime_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision (AP) for pre-crime frame detection.

    y_true: 1 for pre-crime frames, 0 otherwise.
    Computes the area under the precision-recall curve via the standard
    step-interpolated formula: AP = sum_k P(k) * ΔR(k).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n_positive = int(np.sum(y_true))
    if n_positive == 0:
        return 0.0
    sorted_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[sorted_idx]
    tp_cumsum = np.cumsum(y_sorted).astype(float)
    precision_at_k = tp_cumsum / (np.arange(len(y_true), dtype=float) + 1.0)
    # sum precision only at positions where a positive is retrieved
    return float(np.sum(precision_at_k[y_sorted == 1]) / n_positive)
