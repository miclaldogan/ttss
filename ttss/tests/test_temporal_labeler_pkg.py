"""Temporal Threat Scoring System (TTSS): package tests for temporal labeling."""

from ttss.data.temporal_labeler import (
    CRIME_LABEL,
    NORMAL_LABEL,
    POST_CRIME_LABEL,
    PRE_CRIME_LABEL,
    TemporalSpan,
    TemporalThreatLabeler,
)


def test_temporal_labeler_uses_default_90_frame_context() -> None:
    labeler = TemporalThreatLabeler()

    boundaries = labeler.compute_phase_ranges(100, 120, total_frames=260)

    assert boundaries[PRE_CRIME_LABEL] == TemporalSpan(10, 99)
    assert boundaries[CRIME_LABEL] == TemporalSpan(100, 120)
    assert boundaries[POST_CRIME_LABEL] == TemporalSpan(121, 210)


def test_temporal_labeler_generates_expected_labels_and_scores() -> None:
    labeler = TemporalThreatLabeler(pre_window=90, post_window=90)

    pre_label = labeler.label_frame(99, 100, 120, total_frames=260)
    crime_label = labeler.label_frame(120, 100, 120, total_frames=260)
    post_label = labeler.label_frame(121, 100, 120, total_frames=260)
    normal_label = labeler.label_frame(0, 100, 120, total_frames=260)

    assert pre_label.label == PRE_CRIME_LABEL
    assert 0.4 < pre_label.threat_score <= 0.5
    assert crime_label.label == CRIME_LABEL
    assert crime_label.threat_score == 1.0
    assert post_label.label == POST_CRIME_LABEL
    assert 0.0 < post_label.threat_score < 1.0
    assert normal_label.label == NORMAL_LABEL
    assert normal_label.threat_score == 0.0


def test_temporal_labeler_merges_multiple_spans_by_max_score() -> None:
    labeler = TemporalThreatLabeler(pre_window=10, post_window=10)

    labels = labeler.label_intervals(
        total_frames=60,
        anomaly_spans=[TemporalSpan(20, 25), TemporalSpan(30, 35)],
    )

    assert labels[19].label == PRE_CRIME_LABEL
    assert labels[20].label == CRIME_LABEL
    assert labels[26].label == POST_CRIME_LABEL
    assert labels[29].label == PRE_CRIME_LABEL
