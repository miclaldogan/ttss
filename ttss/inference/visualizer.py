"""Temporal Threat Scoring System (TTSS): threat score visualization.

Four outputs:

1. overlay()             — draw threat score bar + label on a single frame (cv2).
2. plot_score_timeline() — matplotlib threat score over time with crime onset marker.
3. plot_attention_heatmap() — BiLSTM attention weights as a heatmap over the timeline.
4. plot_roc_curve()      — ROC curve with AUC annotation.
5. plot_ear_curve()      — Early Alert Rate curve across thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(slots=True)
class ThreatScoreVisualizer:
    """Overlay helper: draw a threat score bar and label on a BGR frame."""

    color: tuple[int, int, int] = (0, 0, 255)     # red in BGR
    bar_height: int = 20
    font_scale: float = 0.6
    thickness: int = 2

    def overlay(self, frame: Any, score: float, label: str = "") -> Any:
        """Draw score bar and label onto *frame* in-place and return it.

        Falls back to returning the unmodified frame when cv2 is not available.
        """
        try:
            import cv2
        except ImportError:
            return frame

        h, w = frame.shape[:2]
        bar_w = int(w * max(0.0, min(1.0, score)))
        cv2.rectangle(frame, (0, h - self.bar_height), (bar_w, h), self.color, -1)
        cv2.rectangle(frame, (0, h - self.bar_height), (w, h), (200, 200, 200), 1)

        text = f"{label} {score:.2f}" if label else f"{score:.2f}"
        cv2.putText(
            frame, text, (4, h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.thickness,
        )
        return frame


# ---------------------------------------------------------------------------
# Score timeline
# ---------------------------------------------------------------------------


def plot_score_timeline(
    scores: Sequence[float],
    frame_labels: Sequence[int] | None = None,
    crime_start: int | None = None,
    fps: float = 30.0,
    frame_stride: int = 1,
    threshold: float = 0.5,
    title: str = "Threat Score Timeline",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Plot per-frame threat scores as a line chart.

    Parameters
    ----------
    scores:       Per-frame threat scores in [0, 1].
    frame_labels: Optional binary ground-truth (1 = crime, 0 = normal).
    crime_start:  Frame index of crime onset — draws a vertical red line.
    fps:          Frame rate for x-axis conversion to seconds.
    frame_stride: Stride used during feature extraction (scales x-axis).
    threshold:    Decision threshold — draws a horizontal dashed line.
    title:        Plot title.
    save_path:    If given, save the figure to this path.
    show:         If True, call ``plt.show()``.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt
    import numpy as np

    scores_arr = np.asarray(scores, dtype=float)
    T = len(scores_arr)
    times = np.arange(T) * frame_stride / fps

    fig, ax = plt.subplots(figsize=(12, 4))

    if frame_labels is not None:
        labels_arr = np.asarray(frame_labels)
        ax.fill_between(
            times, 0, 1,
            where=labels_arr == 1,
            alpha=0.15, color="red", label="Crime frames",
        )

    ax.plot(times, scores_arr, color="#2196F3", linewidth=1.5, label="Threat score")
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.0, label=f"Threshold {threshold:.2f}")

    if crime_start is not None:
        t_onset = crime_start * frame_stride / fps
        ax.axvline(t_onset, color="red", linewidth=1.5, linestyle="-", label=f"Crime onset ({t_onset:.1f}s)")

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Threat score")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------


def plot_attention_heatmap(
    attention_weights: Sequence[float],
    scores: Sequence[float] | None = None,
    crime_start: int | None = None,
    fps: float = 30.0,
    frame_stride: int = 1,
    title: str = "BiLSTM Attention Weights",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Plot BiLSTM temporal attention weights as a heatmap strip.

    Attention weights show which frames the model focused on most when making
    its threat prediction.  High attention + high score = the model is
    confidently flagging those frames.

    Parameters
    ----------
    attention_weights: Per-frame attention weights (sum to 1.0).
    scores:            Optional per-frame threat scores for second subplot.
    crime_start:       Crime onset frame index.
    fps, frame_stride: Used for time-axis scaling.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    attn = np.asarray(attention_weights, dtype=float)
    T = len(attn)
    times = np.arange(T) * frame_stride / fps

    n_rows = 2 if scores is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2 * n_rows + 1), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # Attention heatmap row
    ax_attn = axes[0]
    ax_attn.imshow(
        attn[np.newaxis, :], aspect="auto", cmap="YlOrRd",
        extent=[times[0], times[-1], 0, 1],
    )
    if crime_start is not None:
        ax_attn.axvline(crime_start * frame_stride / fps, color="blue", linewidth=1.5)
    ax_attn.set_yticks([])
    ax_attn.set_ylabel("Attention", fontsize=9)
    ax_attn.set_title(title)

    if scores is not None:
        ax_score = axes[1]
        ax_score.plot(times, np.asarray(scores, dtype=float), color="#2196F3", linewidth=1.2)
        if crime_start is not None:
            ax_score.axvline(crime_start * frame_stride / fps, color="red", linewidth=1.5)
        ax_score.set_ylim(-0.05, 1.05)
        ax_score.set_ylabel("Threat score", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    fpr: Sequence[float],
    tpr: Sequence[float],
    auc: float,
    label: str = "TTSS",
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
    show: bool = False,
    extra_curves: list[dict] | None = None,
) -> Any:
    """Plot a ROC curve with AUC annotation.

    Parameters
    ----------
    fpr, tpr:     ROC curve arrays from an evaluator.
    auc:          Area under the curve.
    extra_curves: List of ``{"fpr": ..., "tpr": ..., "auc": ..., "label": ...}``
                  dicts for comparing multiple models on the same axes.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.4f})")

    if extra_curves:
        for curve in extra_curves:
            ax.plot(
                curve["fpr"], curve["tpr"], linewidth=1.5, linestyle="--",
                label=f"{curve.get('label', '?')} (AUC={curve.get('auc', 0):.4f})",
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# EAR curve
# ---------------------------------------------------------------------------


def plot_ear_curve(
    thresholds: Sequence[float],
    ear_values: Sequence[float],
    title: str = "Early Alert Rate Curve",
    save_path: str | Path | None = None,
    show: bool = False,
    extra_curves: list[dict] | None = None,
) -> Any:
    """Plot the Early Alert Rate (EAR) vs. threshold curve.

    EAR at threshold t = fraction of crime videos where at least one
    pre-crime frame scores ≥ t.  The curve shows the trade-off between
    alert sensitivity and false-alarm rate as the threshold is tightened.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, ear_values, marker="o", markersize=4, linewidth=2, label="TTSS")

    if extra_curves:
        for curve in extra_curves:
            ax.plot(
                curve["thresholds"], curve["ear_values"],
                marker="s", markersize=3, linewidth=1.5, linestyle="--",
                label=curve.get("label", "?"),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Early Alert Rate (EAR)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
