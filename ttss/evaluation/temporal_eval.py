"""Temporal Threat Scoring System (TTSS): temporal evaluation routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


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
