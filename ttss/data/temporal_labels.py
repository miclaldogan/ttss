"""Temporal Threat Scoring System (TTSS): temporal label generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class TemporalLabel(str, Enum):
    """Label names used by the TTSS temporal annotation scheme."""

    BACKGROUND = "background"
    PRE_CRIME = "pre-crime"
    CRIME = "crime"
    POST_CRIME = "post-crime"


@dataclass(frozen=True, init=False)
class TemporalWindowConfig:
    """Configuration for pre-crime and post-crime temporal windows."""

    pre_window: int
    post_window: int

    def __init__(
        self,
        pre_window: int = 30,
        post_window: int = 30,
        *,
        pre_crime_frames: int | None = None,
        post_crime_frames: int | None = None,
    ) -> None:
        resolved_pre = pre_crime_frames if pre_crime_frames is not None else pre_window
        resolved_post = (
            post_crime_frames if post_crime_frames is not None else post_window
        )
        if resolved_pre < 0 or resolved_post < 0:
            raise ValueError("Temporal windows must be non-negative")

        object.__setattr__(self, "pre_window", resolved_pre)
        object.__setattr__(self, "post_window", resolved_post)

    @property
    def pre_crime_frames(self) -> int:
        """Backward-compatible alias for the pre-crime window size."""

        return self.pre_window

    @property
    def post_crime_frames(self) -> int:
        """Backward-compatible alias for the post-crime window size."""

        return self.post_window


def compute_temporal_boundaries(
    crime_start_frame: int,
    crime_end_frame: int,
    config: TemporalWindowConfig | None = None,
) -> dict[str, tuple[int, int]]:
    """Compute pre-crime, crime, and post-crime frame intervals."""
    if crime_start_frame < 0:
        raise ValueError("crime_start_frame must be non-negative")
    if crime_end_frame < crime_start_frame:
        raise ValueError("crime_end_frame must be >= crime_start_frame")

    windows = config or TemporalWindowConfig()
    pre_start = max(0, crime_start_frame - windows.pre_window)
    post_end = crime_end_frame + windows.post_window

    return {
        TemporalLabel.PRE_CRIME.value: (pre_start, crime_start_frame - 1),
        TemporalLabel.CRIME.value: (crime_start_frame, crime_end_frame),
        TemporalLabel.POST_CRIME.value: (crime_end_frame + 1, post_end),
    }


def assign_temporal_label(
    frame_index: int,
    crime_start_frame: int,
    crime_end_frame: int,
    config: TemporalWindowConfig | None = None,
) -> str:
    """Assign a TTSS temporal label to a single frame index."""
    windows = config or TemporalWindowConfig()
    boundaries = compute_temporal_boundaries(
        crime_start_frame=crime_start_frame,
        crime_end_frame=crime_end_frame,
        config=windows,
    )
    pre_start, pre_end = boundaries[TemporalLabel.PRE_CRIME.value]
    crime_start, crime_end = boundaries[TemporalLabel.CRIME.value]
    post_start, post_end = boundaries[TemporalLabel.POST_CRIME.value]

    if pre_start <= frame_index <= pre_end:
        return TemporalLabel.PRE_CRIME.value
    if crime_start <= frame_index <= crime_end:
        return TemporalLabel.CRIME.value
    if post_start <= frame_index <= post_end:
        return TemporalLabel.POST_CRIME.value
    return TemporalLabel.BACKGROUND.value


def label_frames(
    frame_indices: Iterable[int],
    crime_start_frame: int,
    crime_end_frame: int,
    config: TemporalWindowConfig | None = None,
) -> list[str]:
    """Assign TTSS temporal labels to a sequence of frames."""
    return [
        assign_temporal_label(
            frame_index=frame_index,
            crime_start_frame=crime_start_frame,
            crime_end_frame=crime_end_frame,
            config=config,
        )
        for frame_index in frame_indices
    ]
