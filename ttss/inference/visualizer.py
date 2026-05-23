"""Temporal Threat Scoring System (TTSS): threat score visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ThreatScoreVisualizer:
    """Overlay helper for visualizing threat scores on frames."""

    color: tuple[int, int, int] = (255, 0, 0)

    def overlay(self, frame: Any, score: float, label: str) -> Any:
        """Return the input frame until a drawing backend is connected."""
        del score, label
        return frame
