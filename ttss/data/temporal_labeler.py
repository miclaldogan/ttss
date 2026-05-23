"""Temporal Threat Scoring System (TTSS): temporal tagging and threat scoring.

This module implements the TTSS temporal labeling policy for UCF-Crime video
annotations. It converts anomaly intervals into per-frame tags and continuous
threat scores spanning normal, pre-crime, crime, and post-crime phases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


NORMAL_LABEL = "normal"
PRE_CRIME_LABEL = "pre_crime"
CRIME_LABEL = "crime"
POST_CRIME_LABEL = "post_crime"


@dataclass(frozen=True, slots=True)
class TemporalThreatLabel:
    """Per-frame TTSS temporal label and continuous threat score."""

    frame_index: int
    label: str
    threat_score: float


@dataclass(frozen=True, slots=True)
class TemporalSpan:
    """Closed frame interval for a temporal event segment."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")


@dataclass(frozen=True, slots=True)
class TemporalThreatLabeler:
    """Custom temporal labeling scheme.

    Label scheme:
    - pre_crime: K frames before the crime starts
    - crime: frames inside the anomaly interval
    - post_crime: K frames after the crime ends
    - normal: all remaining frames

    Threat score:
    - normal -> 0.0
    - pre_crime -> linear increase from 0.0 to 0.5
    - crime -> density-based increase from 0.5 to 1.0
    - post_crime -> exponential decrease from 1.0 to 0.0
    """

    pre_window: int = 90
    post_window: int = 90
    post_decay: float = 5.0

    def __post_init__(self) -> None:
        if self.pre_window < 0 or self.post_window < 0:
            raise ValueError("Temporal windows must be non-negative")
        if self.post_decay <= 0.0:
            raise ValueError("post_decay must be positive")

    def compute_phase_ranges(
        self,
        crime_start_frame: int,
        crime_end_frame: int,
        total_frames: int | None = None,
    ) -> dict[str, TemporalSpan]:
        """Compute closed frame intervals for pre/crime/post segments."""
        if crime_start_frame < 0:
            raise ValueError("crime_start_frame must be non-negative")
        if crime_end_frame < crime_start_frame:
            raise ValueError("crime_end_frame must be >= crime_start_frame")

        last_frame = None if total_frames is None else max(total_frames - 1, 0)
        pre_start = max(0, crime_start_frame - self.pre_window)
        post_end = crime_end_frame + self.post_window
        if last_frame is not None:
            post_end = min(post_end, last_frame)

        return {
            PRE_CRIME_LABEL: TemporalSpan(pre_start, max(pre_start, crime_start_frame - 1)),
            CRIME_LABEL: TemporalSpan(crime_start_frame, crime_end_frame),
            POST_CRIME_LABEL: TemporalSpan(
                min(crime_end_frame + 1, post_end),
                post_end,
            ),
        }

    def label_frame(
        self,
        frame_index: int,
        crime_start_frame: int,
        crime_end_frame: int,
        total_frames: int | None = None,
    ) -> TemporalThreatLabel:
        """Assign one TTSS temporal label and threat score to a frame."""
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")

        ranges = self.compute_phase_ranges(
            crime_start_frame=crime_start_frame,
            crime_end_frame=crime_end_frame,
            total_frames=total_frames,
        )

        pre_range = ranges[PRE_CRIME_LABEL]
        crime_range = ranges[CRIME_LABEL]
        post_range = ranges[POST_CRIME_LABEL]

        if pre_range.start_frame <= frame_index <= pre_range.end_frame:
            return TemporalThreatLabel(
                frame_index=frame_index,
                label=PRE_CRIME_LABEL,
                threat_score=self._pre_crime_score(frame_index, crime_start_frame),
            )

        if crime_range.start_frame <= frame_index <= crime_range.end_frame:
            return TemporalThreatLabel(
                frame_index=frame_index,
                label=CRIME_LABEL,
                threat_score=self._crime_score(
                    frame_index=frame_index,
                    crime_start_frame=crime_start_frame,
                    crime_end_frame=crime_end_frame,
                ),
            )

        if post_range.start_frame <= frame_index <= post_range.end_frame:
            return TemporalThreatLabel(
                frame_index=frame_index,
                label=POST_CRIME_LABEL,
                threat_score=self._post_crime_score(frame_index, crime_end_frame),
            )

        return TemporalThreatLabel(
            frame_index=frame_index,
            label=NORMAL_LABEL,
            threat_score=0.0,
        )

    def label_video(
        self,
        total_frames: int,
        crime_start_frame: int,
        crime_end_frame: int,
        frame_indices: Iterable[int] | None = None,
    ) -> list[TemporalThreatLabel]:
        """Generate per-frame TTSS labels and scores for a full video or subset."""
        if total_frames <= 0:
            raise ValueError("total_frames must be positive")

        indices = list(range(total_frames)) if frame_indices is None else list(frame_indices)
        return [
            self.label_frame(
                frame_index=frame_index,
                crime_start_frame=crime_start_frame,
                crime_end_frame=crime_end_frame,
                total_frames=total_frames,
            )
            for frame_index in indices
        ]

    def build_annotation_payload(
        self,
        video_id: str,
        total_frames: int,
        crime_start_frame: int,
        crime_end_frame: int,
        label: str,
        fps: float = 30.0,
        include_frame_labels: bool = True,
    ) -> dict[str, object]:
        """Serialize temporal tags into an annotation JSON payload."""
        frame_labels = self.label_video(
            total_frames=total_frames,
            crime_start_frame=crime_start_frame,
            crime_end_frame=crime_end_frame,
        )
        ranges = self.compute_phase_ranges(
            crime_start_frame=crime_start_frame,
            crime_end_frame=crime_end_frame,
            total_frames=total_frames,
        )
        payload: dict[str, object] = {
            "video_id": video_id,
            "label": label,
            "fps": fps,
            "total_frames": total_frames,
            "segments": {
                PRE_CRIME_LABEL: self._span_to_dict(ranges[PRE_CRIME_LABEL]),
                CRIME_LABEL: self._span_to_dict(ranges[CRIME_LABEL]),
                POST_CRIME_LABEL: self._span_to_dict(ranges[POST_CRIME_LABEL]),
            },
        }
        if include_frame_labels:
            payload["frame_labels"] = [
                {
                    "frame_index": item.frame_index,
                    "label": item.label,
                    "threat_score": round(item.threat_score, 6),
                }
                for item in frame_labels
            ]
        return payload

    def label_intervals(
        self,
        total_frames: int,
        anomaly_spans: Sequence[TemporalSpan],
    ) -> list[TemporalThreatLabel]:
        """Aggregate labels for videos with one or more anomaly intervals."""
        if total_frames <= 0:
            raise ValueError("total_frames must be positive")
        if not anomaly_spans:
            return [
                TemporalThreatLabel(frame_index=index, label=NORMAL_LABEL, threat_score=0.0)
                for index in range(total_frames)
            ]

        labels = [
            TemporalThreatLabel(frame_index=index, label=NORMAL_LABEL, threat_score=0.0)
            for index in range(total_frames)
        ]
        for span in anomaly_spans:
            for item in self.label_video(
                total_frames=total_frames,
                crime_start_frame=span.start_frame,
                crime_end_frame=span.end_frame,
            ):
                existing = labels[item.frame_index]
                if item.threat_score >= existing.threat_score:
                    labels[item.frame_index] = item
        return labels

    def _pre_crime_score(self, frame_index: int, crime_start_frame: int) -> float:
        if self.pre_window == 0:
            return 0.5
        distance = crime_start_frame - frame_index
        normalized = 1.0 - min(max(distance, 0), self.pre_window) / self.pre_window
        return max(0.0, min(0.5, 0.5 * normalized))

    def _crime_score(
        self,
        frame_index: int,
        crime_start_frame: int,
        crime_end_frame: int,
    ) -> float:
        duration = max(crime_end_frame - crime_start_frame + 1, 1)
        density = (frame_index - crime_start_frame + 1) / duration
        return max(0.5, min(1.0, 0.5 + 0.5 * density))

    def _post_crime_score(self, frame_index: int, crime_end_frame: int) -> float:
        if self.post_window == 0:
            return 0.0
        distance = frame_index - crime_end_frame
        normalized = min(max(distance, 0), self.post_window) / self.post_window
        score = math.exp(-self.post_decay * normalized)
        return max(0.0, min(1.0, score))

    def _span_to_dict(self, span: TemporalSpan) -> dict[str, int]:
        return {"start_frame": span.start_frame, "end_frame": span.end_frame}
