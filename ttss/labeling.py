"""Temporal Threat Scoring System (TTSS): compatibility label helpers."""

from ttss.data.temporal_labels import (
    TemporalLabel,
    TemporalWindowConfig,
    assign_temporal_label,
    compute_temporal_boundaries,
    label_frames,
)

__all__ = [
    "TemporalLabel",
    "TemporalWindowConfig",
    "assign_temporal_label",
    "compute_temporal_boundaries",
    "label_frames",
]
