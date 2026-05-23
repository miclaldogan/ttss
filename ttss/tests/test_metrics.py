"""Temporal Threat Scoring System (TTSS): tests for metrics."""

from ttss.training.metrics import binary_f1_score, early_alarm_rate, roc_auc_score


def test_binary_f1_score_is_perfect_for_correct_predictions() -> None:
    assert binary_f1_score([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_roc_auc_score_is_perfect_for_ranked_predictions() -> None:
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]) == 1.0


def test_early_alarm_rate_counts_pre_window_hits() -> None:
    assert early_alarm_rate([0.2, 0.6, 0.8], threshold=0.5) == (2 / 3)
