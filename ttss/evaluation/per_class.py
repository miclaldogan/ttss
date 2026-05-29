"""Temporal Threat Scoring System (TTSS): per-crime-category evaluation.

``PerClassEvaluator`` groups test-set predictions by UCF-Crime category label
and computes AUC, EAR, and MALT per category, producing the table used in the
paper's Table 2 / Figure 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ttss.evaluation.temporal_eval import UCF_CRIME_CATEGORIES
from ttss.training.metrics import early_alert_rate, frame_level_auc, mean_alert_lead_time, precrime_ap


@dataclass
class PerClassResult:
    """Per-category evaluation output.

    Schema matches ``evaluation/per_class_results.json``:
    ``{"category": str, "n_videos": int, "frame_auc": float, "ear": float, "malt_frames": float}``
    """

    category: str
    n_videos: int
    frame_auc: float
    ear: float
    malt_frames: float
    precrime_ap: float = 0.0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "n_videos": self.n_videos,
            "frame_auc": round(self.frame_auc, 4),
            "ear": round(self.ear, 4),
            "malt_frames": round(self.malt_frames, 2),
            "precrime_ap": round(self.precrime_ap, 4),
        }


class PerClassEvaluator:
    """Group predictions by category and compute per-class metrics.

    Usage::

        evaluator = PerClassEvaluator()
        results = evaluator.evaluate(predictions, annotations)

    Parameters
    ----------
    threshold:   Alert decision threshold for EAR and MALT (default 0.5).
    fps:         Frame rate used for MALT seconds conversion (default 30).
    """

    def __init__(self, threshold: float = 0.5, fps: float = 30.0) -> None:
        self.threshold = threshold
        self.fps = fps

    def evaluate(
        self,
        predictions: Sequence[np.ndarray],
        annotations: Sequence[object],
    ) -> list[PerClassResult]:
        """Compute per-category metrics.

        Parameters
        ----------
        predictions:
            Sequence of per-video score arrays, one per video.
        annotations:
            Sequence of objects with ``.label`` (str) and ``.y_true``
            (np.ndarray of 0/1 per frame) attributes.  Compatible with
            ``AnnotationRecord`` when augmented with frame-level labels.

        Returns
        -------
        List of :class:`PerClassResult`, one per category, sorted by
        ``frame_auc`` descending.  The macro-average AUC (weighted by
        n_videos) is appended as a final row with category ``"_macro"``.
        """
        groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for scores, ann in zip(predictions, annotations):
            cat = getattr(ann, "label", "Unknown")
            y_true = np.asarray(getattr(ann, "y_true", np.zeros(len(scores))))
            groups.setdefault(cat, []).append((y_true, np.asarray(scores, dtype=float)))

        results: list[PerClassResult] = []
        total_videos = 0
        weighted_auc = 0.0

        for cat, pairs in groups.items():
            aucs, ears, malts, aps = [], [], [], []
            for y_true, y_score in pairs:
                if len(np.unique(y_true)) >= 2:
                    aucs.append(frame_level_auc(y_true, y_score))
                ears.append(early_alert_rate(y_true, y_score, self.threshold))
                malts.append(mean_alert_lead_time(y_true, y_score, self.threshold))
                onset = int(np.where(y_true == 1)[0][0]) if (y_true == 1).any() else 0
                if onset > 0:
                    pre = np.zeros_like(y_true); pre[:onset] = 1
                    aps.append(precrime_ap(pre, y_score))

            n = len(pairs)
            auc_val = float(np.mean(aucs)) if aucs else 0.0
            results.append(PerClassResult(
                category=cat,
                n_videos=n,
                frame_auc=auc_val,
                ear=float(np.mean(ears)) if ears else 0.0,
                malt_frames=float(np.mean(malts)) if malts else 0.0,
                precrime_ap=float(np.mean(aps)) if aps else 0.0,
            ))
            weighted_auc += auc_val * n
            total_videos += n

        results.sort(key=lambda r: r.frame_auc, reverse=True)

        # Weighted macro average row
        if total_videos > 0:
            results.append(PerClassResult(
                category="_macro",
                n_videos=total_videos,
                frame_auc=weighted_auc / total_videos,
                ear=float(np.mean([r.ear for r in results])),
                malt_frames=float(np.mean([r.malt_frames for r in results])),
            ))

        return results
