"""Temporal Threat Scoring System (TTSS): threat score metrics."""

from __future__ import annotations

from typing import Sequence


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
