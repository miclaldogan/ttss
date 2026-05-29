"""Temporal Threat Scoring System (TTSS): score calibration utilities.

A calibrated model has P(y=1 | score=s) ≈ s.  This module provides:

1. ``reliability_diagram`` — plots observed fraction positive vs. mean predicted
   score in *n_bins* equal-width bins, with ECE and MCE annotations.
2. ``CalibrationResult`` — ECE, MCE, and per-bin data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class CalibrationResult:
    """Output of :func:`reliability_diagram`.

    Attributes
    ----------
    ece:       Expected Calibration Error — weighted mean |confidence - accuracy|.
    mce:       Maximum Calibration Error — worst-bin |confidence - accuracy|.
    bin_confs: Mean predicted score per bin.
    bin_accs:  Observed fraction positive per bin.
    bin_sizes: Number of samples per bin.
    """

    ece: float
    mce: float
    bin_confs: np.ndarray
    bin_accs: np.ndarray
    bin_sizes: np.ndarray

    def __str__(self) -> str:
        return f"ECE={self.ece:.4f}  MCE={self.mce:.4f}"


def reliability_diagram(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    n_bins: int = 10,
    save_path: str | Path | None = None,
    title: str = "Reliability Diagram",
    show: bool = False,
) -> CalibrationResult:
    """Compute ECE / MCE and plot the reliability diagram.

    Parameters
    ----------
    y_true:    Binary ground-truth labels (0 or 1).
    y_score:   Predicted scores in [0, 1].
    n_bins:    Number of equal-width bins (default 10).
    save_path: If given, save figure at 300 DPI to this path.
    title:     Plot title.
    show:      If True, call ``plt.show()``.

    Returns
    -------
    :class:`CalibrationResult` with ECE, MCE, and per-bin arrays.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1])  # 0 … n_bins-1

    bin_confs = np.zeros(n_bins)
    bin_accs = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_confs[b] = float(np.mean(y_score[mask]))
            bin_accs[b] = float(np.mean(y_true[mask]))
            bin_sizes[b] = int(mask.sum())

    total = max(1, len(y_true))
    ece = float(np.sum(bin_sizes * np.abs(bin_accs - bin_confs)) / total)
    gaps = np.abs(bin_accs - bin_confs)
    mce = float(gaps[bin_sizes > 0].max()) if (bin_sizes > 0).any() else 0.0

    result = CalibrationResult(
        ece=ece, mce=mce,
        bin_confs=bin_confs, bin_accs=bin_accs, bin_sizes=bin_sizes,
    )

    if save_path is not None or show:
        _plot_reliability(result, n_bins, bins, title, save_path, show)

    return result


def _plot_reliability(
    result: CalibrationResult,
    n_bins: int,
    bins: np.ndarray,
    title: str,
    save_path,
    show: bool,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_width = 1.0 / n_bins
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(
        bin_centers, result.bin_accs, width=bin_width * 0.9,
        color="#2196F3", alpha=0.7, label="Model",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{title}\nECE={result.ece:.4f}  MCE={result.mce:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
