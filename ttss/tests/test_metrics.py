"""Temporal Threat Scoring System (TTSS): tests for metrics."""

import numpy as np
import pytest

from ttss.training.metrics import (
    binary_f1_score,
    early_alarm_rate,
    early_alert_rate,
    frame_level_auc,
    mean_alert_lead_time,
    precrime_ap,
    roc_auc_score,
)


def test_binary_f1_score_is_perfect_for_correct_predictions() -> None:
    assert binary_f1_score([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_roc_auc_score_is_perfect_for_ranked_predictions() -> None:
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]) == 1.0


def test_early_alarm_rate_counts_pre_window_hits() -> None:
    assert early_alarm_rate([0.2, 0.6, 0.8], threshold=0.5) == (2 / 3)


# ---------------------------------------------------------------------------
# frame_level_auc
# ---------------------------------------------------------------------------


def test_frame_level_auc_perfect_ranking() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert frame_level_auc(y_true, y_score) == pytest.approx(1.0)


def test_frame_level_auc_worst_ranking() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.8, 0.9, 0.1, 0.2])
    assert frame_level_auc(y_true, y_score) == pytest.approx(0.0)


def test_frame_level_auc_tie_is_half() -> None:
    y_true = np.array([0, 1])
    y_score = np.array([0.5, 0.5])
    assert frame_level_auc(y_true, y_score) == pytest.approx(0.5)


def test_frame_level_auc_no_positives_returns_zero() -> None:
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.9])
    assert frame_level_auc(y_true, y_score) == 0.0


def test_frame_level_auc_no_negatives_returns_zero() -> None:
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.8, 0.7, 0.6])
    assert frame_level_auc(y_true, y_score) == 0.0


# ---------------------------------------------------------------------------
# early_alert_rate
# ---------------------------------------------------------------------------


def test_early_alert_rate_all_fire() -> None:
    # onset at index 4; all pre-crime frames exceed threshold
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.8, 0.7, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert early_alert_rate(y_true, y_score, threshold=0.5) == pytest.approx(1.0)


def test_early_alert_rate_none_fire() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
    assert early_alert_rate(y_true, y_score, threshold=0.5) == pytest.approx(0.0)


def test_early_alert_rate_partial() -> None:
    # onset=4; frames 0-3 have scores [0.8, 0.1, 0.8, 0.1] → 2/4 = 0.5
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.8, 0.1, 0.8, 0.1, 0.9, 0.9])
    assert early_alert_rate(y_true, y_score, threshold=0.5) == pytest.approx(0.5)


def test_early_alert_rate_no_crime_returns_zero() -> None:
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.8, 0.8, 0.8])
    assert early_alert_rate(y_true, y_score) == 0.0


def test_early_alert_rate_onset_at_zero_returns_zero() -> None:
    # crime starts at frame 0 → no pre-crime window
    y_true = np.array([1, 1, 1, 0])
    y_score = np.array([0.9, 0.9, 0.9, 0.1])
    assert early_alert_rate(y_true, y_score) == 0.0


# ---------------------------------------------------------------------------
# mean_alert_lead_time
# ---------------------------------------------------------------------------


def test_mean_alert_lead_time_correct() -> None:
    # onset=5; first alert fires at index 2 → lead time=3
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9])
    assert mean_alert_lead_time(y_true, y_score, threshold=0.5) == pytest.approx(3.0)


def test_mean_alert_lead_time_no_alert_returns_zero() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
    assert mean_alert_lead_time(y_true, y_score, threshold=0.5) == pytest.approx(0.0)


def test_mean_alert_lead_time_no_crime_returns_zero() -> None:
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.9, 0.9, 0.9])
    assert mean_alert_lead_time(y_true, y_score) == 0.0


# ---------------------------------------------------------------------------
# precrime_ap
# ---------------------------------------------------------------------------


def test_precrime_ap_perfect() -> None:
    # pre-crime frames [0,1,2] ranked highest → AP=1.0
    y_true = np.array([1, 1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precrime_ap(y_true, y_score) == pytest.approx(1.0)


def test_precrime_ap_no_positives_returns_zero() -> None:
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.9, 0.5, 0.1])
    assert precrime_ap(y_true, y_score) == 0.0


def test_precrime_ap_between_zero_and_one() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=50)
    y_score = rng.random(50)
    result = precrime_ap(y_true, y_score)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# evaluate_sequence (temporal_eval integration)
# ---------------------------------------------------------------------------


def test_evaluate_sequence_returns_report() -> None:
    from ttss.evaluation.temporal_eval import evaluate_sequence

    y_true_crime = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.2, 0.3, 0.8, 0.7, 0.9, 0.9, 0.9, 0.9])
    report = evaluate_sequence(y_true_crime, y_score)
    assert hasattr(report, "frame_level_auc")
    assert hasattr(report, "early_alert_rate")
    assert hasattr(report, "mean_alert_lead_time")
    assert hasattr(report, "precrime_ap")
    assert 0.0 <= report.frame_level_auc <= 1.0
    assert 0.0 <= report.early_alert_rate <= 1.0
    assert report.mean_alert_lead_time >= 0.0
    assert 0.0 <= report.precrime_ap <= 1.0


# ---------------------------------------------------------------------------
# evaluate CLI
# ---------------------------------------------------------------------------


def test_evaluate_script_parser_accepts_checkpoint_and_output() -> None:
    from ttss.scripts.evaluate import build_parser

    args = build_parser().parse_args(["--checkpoint", "fake.pt", "--output", "out.json"])
    assert args.checkpoint == "fake.pt"
    assert args.output == "out.json"


def test_evaluate_script_runs_without_checkpoint(tmp_path) -> None:
    from ttss.scripts.evaluate import _compute_metrics, _synthetic_sequences

    rng = np.random.default_rng(0)
    seqs = _synthetic_sequences(5, 32, rng)
    metrics = _compute_metrics(seqs, threshold=0.5)
    assert set(metrics) == {
        "frame_level_auc",
        "early_alert_rate",
        "mean_alert_lead_time_frames",
        "precrime_ap",
    }
    for v in metrics.values():
        assert isinstance(v, float)


def test_evaluate_script_writes_json(tmp_path) -> None:
    import json
    import sys

    output_file = tmp_path / "results.json"
    sys.argv = [
        "evaluate",
        "--output", str(output_file),
        "--n-videos", "5",
        "--n-frames", "32",
    ]
    from ttss.scripts.evaluate import main

    main()
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert "metrics" in data
    assert "frame_level_auc" in data["metrics"]

