"""Temporal Threat Scoring System (TTSS): tests for pre-crime metrics (issue #11).

Covers:
  - MALTResult / EARCurveResult / ROCResult datatypes
  - mean_alert_lead_time: no alarm, alarm at onset, alarm before onset, list input
  - ear_curve: monotonically non-increasing, edge cases
  - temporal_roc: T-AUC@L, non-crime videos, empty pre-window
  - evaluate_precrime CLI: writes valid precrime_results.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from ttss.evaluation.precrime_metrics import (
    EARCurveResult,
    MALTResult,
    PreCrimeMetrics,
    ROCResult,
)

pm = PreCrimeMetrics(default_fps=30.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(length: int, onset: int, score_before: float, score_after: float) -> np.ndarray:
    s = np.full(length, 0.1)
    if onset > 0:
        s[:onset] = score_before
    if onset < length:
        s[onset:] = score_after
    return s


# ---------------------------------------------------------------------------
# MALTResult type
# ---------------------------------------------------------------------------


def test_malt_result_fields() -> None:
    r = MALTResult(malt_frames=5.0, malt_seconds=5.0 / 30.0)
    assert r.malt_frames == pytest.approx(5.0)
    assert r.malt_seconds == pytest.approx(5.0 / 30.0)
    assert r.per_video_frames == []


# ---------------------------------------------------------------------------
# mean_alert_lead_time: acceptance criteria
# ---------------------------------------------------------------------------


def test_malt_zero_when_alarm_at_onset() -> None:
    """MALT = 0 when the alarm fires exactly at onset (no pre-crime alarm)."""
    # Score at onset but nothing before onset exceeds threshold.
    scores = [np.array([0.1, 0.1, 0.8, 0.8])]  # onset=2, frames 0-1 are pre-crime
    scores_no_prealarm = [np.array([0.1, 0.1, 0.8, 0.8])]
    result = pm.mean_alert_lead_time(scores_no_prealarm, crime_starts=[2], threshold=0.5)
    # Pre-crime frames: index 0,1 — both below threshold
    assert result.malt_frames == pytest.approx(0.0)
    assert result.malt_seconds == pytest.approx(0.0)


def test_malt_positive_when_alarm_before_onset() -> None:
    """MALT > 0 when alarms fire before crime onset."""
    # onset=10; alarm at frame 5 → lead = 5
    onset = 10
    scores_arr = np.full(20, 0.1)
    scores_arr[5] = 0.9  # alarm at frame 5
    result = pm.mean_alert_lead_time([scores_arr], crime_starts=[onset], threshold=0.5)
    assert result.malt_frames > 0.0
    assert result.per_video_frames[0] == pytest.approx(float(onset - 5))


def test_malt_equals_onset_minus_first_alarm() -> None:
    """MALT = onset - first_alarm_frame for single video."""
    onset = 20
    s = np.zeros(40)
    s[8] = 0.9   # first alarm at frame 8
    s[12] = 0.9  # second alarm
    result = pm.mean_alert_lead_time([s], crime_starts=[onset], threshold=0.5)
    assert result.per_video_frames[0] == pytest.approx(float(onset - 8))


def test_malt_no_alarm_contributes_zero() -> None:
    """Videos with no pre-crime alarm contribute 0 to MALT."""
    onset = 10
    s_no_alarm = np.full(20, 0.1)   # no alarm
    s_alarm = np.full(20, 0.1)
    s_alarm[5] = 0.9               # alarm at frame 5 → lead 5
    result = pm.mean_alert_lead_time(
        [s_no_alarm, s_alarm], crime_starts=[onset, onset], threshold=0.5
    )
    assert result.per_video_frames[0] == pytest.approx(0.0)
    assert result.per_video_frames[1] == pytest.approx(5.0)
    assert result.malt_frames == pytest.approx(2.5)  # (0 + 5) / 2


def test_malt_accepts_list_input() -> None:
    """Accepts list[float] instead of np.ndarray (acceptance criterion)."""
    scores = [[0.9, 0.9, 0.1, 0.1]]  # alarm at frame 0 → lead = 2
    result = pm.mean_alert_lead_time(scores, crime_starts=[2], threshold=0.5)
    assert result.malt_frames == pytest.approx(2.0)


def test_malt_fps_conversion() -> None:
    s = np.zeros(20)
    s[5] = 0.9  # alarm at 5, onset=10, lead=5
    result = pm.mean_alert_lead_time([s], crime_starts=[10], fps=25.0, threshold=0.5)
    assert result.malt_seconds == pytest.approx(5.0 / 25.0)


def test_malt_onset_zero_contributes_zero() -> None:
    """Onset at frame 0 → no pre-crime window → lead = 0."""
    s = np.array([0.9, 0.9, 0.9])
    result = pm.mean_alert_lead_time([s], crime_starts=[0], threshold=0.5)
    assert result.malt_frames == pytest.approx(0.0)


def test_malt_uses_default_fps() -> None:
    s = np.zeros(20)
    s[5] = 0.9
    result = pm.mean_alert_lead_time([s], crime_starts=[10])  # no fps arg
    assert result.malt_seconds == pytest.approx(5.0 / 30.0)


# ---------------------------------------------------------------------------
# EAR curve: acceptance criteria
# ---------------------------------------------------------------------------


def test_ear_curve_returns_correct_type() -> None:
    onset = 10
    scores = [np.full(20, 0.6)]
    result = pm.ear_curve(scores, crime_starts=[onset])
    assert isinstance(result, EARCurveResult)
    assert isinstance(result.thresholds, np.ndarray)
    assert isinstance(result.ear_values, np.ndarray)
    assert len(result.thresholds) == len(result.ear_values)


def test_ear_curve_monotonically_non_increasing() -> None:
    """EAR curve is monotonically non-increasing as threshold increases."""
    rng = np.random.default_rng(0)
    onset = 30
    scores = [rng.random(60) for _ in range(10)]
    result = pm.ear_curve(scores, crime_starts=[onset] * 10)
    # EAR values must be non-increasing.
    assert np.all(np.diff(result.ear_values) <= 1e-9)


def test_ear_curve_high_threshold_low_ear() -> None:
    """At threshold=1.0 (or near), EAR ≈ 0 since scores are <1."""
    onset = 5
    scores = [np.full(10, 0.5)]
    result = pm.ear_curve(scores, crime_starts=[onset], thresholds=[0.9])
    assert result.ear_values[0] == pytest.approx(0.0)


def test_ear_curve_low_threshold_high_ear() -> None:
    """At threshold=0.0, every video with pre-crime frames has EAR=1."""
    onset = 5
    scores = [np.full(10, 0.5)]
    result = pm.ear_curve(scores, crime_starts=[onset], thresholds=[0.0])
    assert result.ear_values[0] == pytest.approx(1.0)


def test_ear_curve_partial_videos_alarmed() -> None:
    """EAR = 0.5 when half the videos alarm."""
    onset = 5
    s_high = np.full(10, 0.9)  # alarm
    s_low = np.full(10, 0.1)   # no alarm
    result = pm.ear_curve([s_high, s_low], crime_starts=[onset, onset], thresholds=[0.5])
    assert result.ear_values[0] == pytest.approx(0.5)


def test_ear_curve_accepts_list_input() -> None:
    """Accepts list[float] score arrays (acceptance criterion)."""
    scores = [[0.9, 0.9, 0.1, 0.1]]
    result = pm.ear_curve(scores, crime_starts=[2], thresholds=[0.5])
    assert result.ear_values[0] == pytest.approx(1.0)


def test_ear_curve_default_thresholds_cover_0_1_to_0_9() -> None:
    scores = [np.full(20, 0.5)]
    result = pm.ear_curve(scores, crime_starts=[10])
    assert result.thresholds.min() == pytest.approx(0.1, abs=0.01)
    assert result.thresholds.max() <= 0.91


def test_ear_curve_empty_scores_returns_zeros() -> None:
    result = pm.ear_curve([], crime_starts=[])
    assert np.all(result.ear_values == 0.0)


# ---------------------------------------------------------------------------
# temporal_roc: acceptance criteria
# ---------------------------------------------------------------------------


def test_temporal_roc_returns_dict_with_default_lead_times() -> None:
    scores = [np.random.default_rng(0).random(30) for _ in range(6)]
    crime_starts: list[int | None] = [15, 20, 10, None, None, None]
    result = pm.temporal_roc(scores, crime_starts)
    assert set(result.keys()) == {0, 30, 60, 90}
    for L, roc in result.items():
        assert isinstance(roc, ROCResult)
        assert isinstance(roc.fpr, np.ndarray)
        assert isinstance(roc.tpr, np.ndarray)
        assert 0.0 <= roc.auc <= 1.0


def test_temporal_roc_perfect_separation() -> None:
    """Perfect model: T-AUC@0 = 1.0 when crime scores >> non-crime scores."""
    crime_scores = [np.full(20, 0.95)]
    noncrime_scores = [np.full(20, 0.05)]
    scores = crime_scores + noncrime_scores
    crime_starts: list[int | None] = [10, None]
    result = pm.temporal_roc(scores, crime_starts, lead_times=[0])
    assert result[0].auc == pytest.approx(1.0)


def test_temporal_roc_no_signal_at_large_lead_time() -> None:
    """When onset <= L, the crime video contributes score=0.0 (no window)."""
    # onset=5 with L=30 → cutoff = max(0, 5-30) = 0 → score = 0.0
    crime_s = np.full(40, 0.9)
    non_crime_s = np.full(40, 0.1)
    scores = [crime_s, non_crime_s]
    crime_starts: list[int | None] = [5, None]
    result = pm.temporal_roc(scores, crime_starts, lead_times=[30])
    # crime score = 0.0, non-crime max = 0.1 → crime is worse than non-crime
    assert result[30].auc <= 0.5


def test_temporal_roc_accepts_list_input() -> None:
    """Accepts list[float] score arrays (acceptance criterion)."""
    scores: list = [[0.9] * 20, [0.1] * 20]
    crime_starts: list[int | None] = [10, None]
    result = pm.temporal_roc(scores, crime_starts, lead_times=[0])
    assert result[0].auc == pytest.approx(1.0)


def test_temporal_roc_custom_lead_times() -> None:
    rng = np.random.default_rng(1)
    scores = [rng.random(50) for _ in range(8)]
    crime_starts: list[int | None] = [25, 30, 20, 35, None, None, None, None]
    result = pm.temporal_roc(scores, crime_starts, lead_times=[0, 15, 45])
    assert set(result.keys()) == {0, 15, 45}


# ---------------------------------------------------------------------------
# evaluate_precrime CLI
# ---------------------------------------------------------------------------


def test_evaluate_precrime_writes_json(tmp_path) -> None:
    from ttss.scripts.evaluate_precrime import main

    out = tmp_path / "precrime_results.json"
    sys.argv = [
        "evaluate_precrime",
        "--output", str(out),
        "--n-videos", "10",
        "--n-frames", "40",
        "--seed", "7",
    ]
    main()
    assert out.exists()
    data = json.loads(out.read_text())
    assert "malt" in data
    assert "ear_curve" in data
    assert "temporal_roc" in data
    assert data["n_videos"] == 10


def test_evaluate_precrime_malt_schema(tmp_path) -> None:
    from ttss.scripts.evaluate_precrime import main

    out = tmp_path / "precrime_results.json"
    sys.argv = ["evaluate_precrime", "--output", str(out)]
    main()
    data = json.loads(out.read_text())
    malt = data["malt"]
    assert "malt_frames" in malt
    assert "malt_seconds" in malt
    assert "per_video_frames" in malt
    assert isinstance(malt["malt_frames"], float)


def test_evaluate_precrime_ear_curve_schema(tmp_path) -> None:
    from ttss.scripts.evaluate_precrime import main

    out = tmp_path / "precrime_results.json"
    sys.argv = ["evaluate_precrime", "--output", str(out)]
    main()
    data = json.loads(out.read_text())
    ear = data["ear_curve"]
    assert len(ear["thresholds"]) == len(ear["ear_values"])
    # Monotonically non-increasing.
    values = ear["ear_values"]
    assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1))


def test_evaluate_precrime_temporal_roc_schema(tmp_path) -> None:
    from ttss.scripts.evaluate_precrime import main

    out = tmp_path / "precrime_results.json"
    sys.argv = ["evaluate_precrime", "--output", str(out)]
    main()
    data = json.loads(out.read_text())
    troc = data["temporal_roc"]
    for L_str, roc in troc.items():
        assert "fpr" in roc
        assert "tpr" in roc
        assert "auc" in roc
        assert 0.0 <= roc["auc"] <= 1.0
