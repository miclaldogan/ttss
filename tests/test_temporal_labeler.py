"""Temporal Threat Scoring System (TTSS): root tests for temporal labeling."""

from __future__ import annotations

import pytest

from ttss.data.temporal_labeler import CRIME_LABEL, PRE_CRIME_LABEL, TemporalThreatLabeler


def test_normal_video_frames_have_zero_threat_score() -> None:
    labeler = TemporalThreatLabeler(pre_window=30, post_window=30)

    labels = labeler.label_intervals(total_frames=120, anomaly_spans=[])

    assert len(labels) == 120
    assert all(item.threat_score == 0.0 for item in labels)


def test_crime_window_scores_are_at_least_half() -> None:
    labeler = TemporalThreatLabeler(pre_window=30, post_window=30)

    labels = labeler.label_video(total_frames=120, crime_start_frame=40, crime_end_frame=60)

    crime_scores = [item.threat_score for item in labels[40:61]]
    assert crime_scores
    assert all(score >= 0.5 for score in crime_scores)


def test_pre_crime_window_has_correct_width() -> None:
    labeler = TemporalThreatLabeler(pre_window=90, post_window=30)

    boundaries = labeler.compute_phase_ranges(120, 150, total_frames=260)

    pre_window = boundaries[PRE_CRIME_LABEL]
    assert pre_window.start_frame == 30
    assert pre_window.end_frame == 119
    assert (pre_window.end_frame - pre_window.start_frame + 1) == 90


def test_all_scores_stay_in_unit_interval() -> None:
    labeler = TemporalThreatLabeler(pre_window=20, post_window=20)

    labels = labeler.label_video(total_frames=100, crime_start_frame=30, crime_end_frame=50)

    assert labels
    assert all(0.0 <= item.threat_score <= 1.0 for item in labels)


def test_pre_crime_scores_increase_linearly() -> None:
    labeler = TemporalThreatLabeler(pre_window=10, post_window=10)

    labels = labeler.label_video(total_frames=40, crime_start_frame=10, crime_end_frame=15)
    pre_crime_labels = [item for item in labels[:10] if item.label == PRE_CRIME_LABEL]

    assert len(pre_crime_labels) == 10
    deltas = [
        right.threat_score - left.threat_score
        for left, right in zip(pre_crime_labels, pre_crime_labels[1:])
    ]
    assert deltas
    assert all(delta > 0.0 for delta in deltas)
    assert all(delta == pytest.approx(deltas[0]) for delta in deltas[1:])
    assert labels[10].label == CRIME_LABEL
