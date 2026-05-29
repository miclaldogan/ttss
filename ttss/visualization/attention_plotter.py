"""TTSS: AttentionHeatmapPlotter — seaborn heatmap of BiLSTM attention weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AttentionHeatmapPlotter:
    """Seaborn-style heatmap of BiLSTM attention weights vs. frame index.

    Usage::

        plotter = AttentionHeatmapPlotter()
        plotter.plot(attention_weights, frame_ids, save_path="figures/attn.png")
    """

    cmap: str = "YlOrRd"
    dpi: int = 300
    figsize: tuple[float, float] = (12, 3)

    def plot(
        self,
        attention_weights: Sequence[float] | Any,
        frame_ids: Sequence[int] | None = None,
        scores: Sequence[float] | None = None,
        crime_start: int | None = None,
        fps: float = 30.0,
        frame_stride: int = 1,
        title: str = "BiLSTM Attention Weights",
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> "plt.Figure":
        """Plot attention weights as a heatmap strip with optional score subplot.

        Parameters
        ----------
        attention_weights: 1-D array of per-frame attention weights (sum ≈ 1).
        frame_ids:         Original frame indices (for x-axis time labels).
        scores:            Optional per-frame threat scores for a second subplot.
        crime_start:       Crime onset frame index → blue vertical line.
        fps, frame_stride: Used for time-axis scaling.
        save_path:         Save at ``self.dpi`` DPI.
        """
        if hasattr(attention_weights, "detach"):
            attn = attention_weights.detach().cpu().numpy().reshape(-1)
        else:
            attn = np.asarray(attention_weights, dtype=float).reshape(-1)

        T = len(attn)
        if frame_ids is not None:
            times = np.array(frame_ids) * frame_stride / fps
        else:
            times = np.arange(T) * frame_stride / fps

        n_rows = 2 if scores is not None else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(self.figsize[0], self.figsize[1] * n_rows),
                                 sharex=True)
        if n_rows == 1:
            axes = [axes]

        # Heatmap row
        ax_h = axes[0]
        extent = [times[0], times[-1], 0, 1]
        ax_h.imshow(attn[np.newaxis, :], aspect="auto", cmap=self.cmap,
                    extent=extent, vmin=0)
        if crime_start is not None:
            ax_h.axvline(crime_start * frame_stride / fps, color="#1565c0", linewidth=1.5)
        ax_h.set_yticks([])
        ax_h.set_ylabel("Attention", fontsize=9)
        ax_h.set_title(title)

        if scores is not None:
            ax_s = axes[1]
            scores_arr = np.asarray(scores, dtype=float)
            ax_s.plot(times[:len(scores_arr)], scores_arr, color="#1565c0", linewidth=1.2)
            if crime_start is not None:
                ax_s.axvline(crime_start * frame_stride / fps, color="#d32f2f", linewidth=1.5)
            ax_s.set_ylim(-0.05, 1.05)
            ax_s.set_ylabel("Threat score", fontsize=9)

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig
