"""TTSS: ThreatScorePlotter — phase-shaded score timeline for the paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Paper color scheme (configurable)
PHASE_COLORS = {
    "normal":    "#e8f5e9",   # light green
    "pre_crime": "#fff9c4",   # light yellow
    "crime":     "#ffcdd2",   # light red
    "post_crime": "#ffe0b2",  # light orange
}
SCORE_COLOR = "#1565c0"
ALARM_COLOR = "#d32f2f"


@dataclass
class ThreatScorePlotter:
    """Matplotlib timeline with four-phase shading for TTSS score trajectories.

    Phases
    ------
    normal     → pre_crime_start
    pre_crime  → pre_crime_start .. crime_start
    crime      → crime_start .. crime_end
    post_crime → crime_end .. end

    Usage::

        plotter = ThreatScorePlotter()
        plotter.plot_timeline(timeline, save_path="figures/robbery.png")
    """

    phase_colors: dict[str, str] = field(default_factory=lambda: dict(PHASE_COLORS))
    score_color: str = SCORE_COLOR
    alarm_color: str = ALARM_COLOR
    dpi: int = 300
    figsize: tuple[float, float] = (12, 4)
    threshold: float = 0.5

    def plot_timeline(
        self,
        timeline: Any,
        crime_start: int | None = None,
        crime_end: int | None = None,
        pre_crime_start: int | None = None,
        fps: float = 30.0,
        frame_stride: int = 1,
        title: str = "Threat Score Timeline",
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> "plt.Figure":
        """Plot a ThreatTimeline (or plain score list) with phase shading.

        Parameters
        ----------
        timeline:         :class:`ThreatTimeline` or ``Sequence[float]`` of scores.
        crime_start:      Crime onset frame index (in original fps).
        crime_end:        Crime end frame index.
        pre_crime_start:  Start of pre-crime window (defaults to crime_start - 90).
        fps, frame_stride: Used for x-axis time conversion.
        save_path:        Save to this path at ``self.dpi`` DPI.
        """
        if hasattr(timeline, "threat_scores"):
            scores = timeline.threat_scores
            frame_ids = timeline.frame_ids
        else:
            scores = list(timeline)
            frame_ids = list(range(len(scores)))

        scores_arr = np.asarray(scores, dtype=float)
        times = np.array(frame_ids) * frame_stride / fps
        T_max = times[-1] if len(times) > 0 else 1.0

        # Convert frame indices to time
        def _t(idx):
            return idx * frame_stride / fps if idx is not None else None

        cs = _t(crime_start)
        ce = _t(crime_end) if crime_end is not None else T_max
        pcs = _t(pre_crime_start) if pre_crime_start is not None else (
            _t(crime_start - 90) if crime_start is not None else None
        )
        pcs = max(0.0, pcs) if pcs is not None else 0.0

        fig, ax = plt.subplots(figsize=self.figsize)

        # Phase shading
        ax.axvspan(0, T_max, alpha=0.25, color=self.phase_colors["normal"], zorder=0)
        if pcs is not None and cs is not None:
            ax.axvspan(pcs, cs, alpha=0.35, color=self.phase_colors["pre_crime"], zorder=1)
        if cs is not None:
            ax.axvspan(cs, ce, alpha=0.35, color=self.phase_colors["crime"], zorder=1)
            if ce < T_max:
                ax.axvspan(ce, T_max, alpha=0.35, color=self.phase_colors["post_crime"], zorder=1)

        ax.plot(times, scores_arr, color=self.score_color, linewidth=1.8, zorder=3, label="Threat score")
        ax.axhline(self.threshold, color=self.alarm_color, linestyle="--",
                   linewidth=1.0, zorder=2, label=f"Threshold {self.threshold:.2f}")

        if cs is not None:
            ax.axvline(cs, color=self.alarm_color, linewidth=1.5, zorder=4)

        # Legend patches for phases
        patches = [
            mpatches.Patch(color=self.phase_colors["normal"], alpha=0.5, label="Normal"),
            mpatches.Patch(color=self.phase_colors["pre_crime"], alpha=0.6, label="Pre-crime"),
            mpatches.Patch(color=self.phase_colors["crime"], alpha=0.6, label="Crime"),
            mpatches.Patch(color=self.phase_colors["post_crime"], alpha=0.6, label="Post-crime"),
        ]
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + patches, fontsize=7, loc="upper left", ncol=3)

        ax.set_xlim(0, T_max)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Threat score")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig
