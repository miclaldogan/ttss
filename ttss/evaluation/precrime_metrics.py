"""Temporal Threat Scoring System (TTSS): pre-crime lead-time metrics.

This module implements the three core novel metrics for the pre-crime detection
contribution:

  1. Mean Alert Lead Time (MALT) — average frames/seconds between the first
     pre-crime alarm and the crime onset.
  2. Early Alert Rate (EAR) Curve — EAR vs. threshold ∈ [0.1, 0.9], showing
     pre-crime sensitivity at every operating point.
  3. Temporal ROC — AUC conditioned on detection at lead-time ≥ L frames,
     T-AUC@L for L ∈ {0, 30, 60, 90}.

All metric functions accept ``np.ndarray`` or ``list[float]`` inputs.

``evaluation/precrime_results.json`` schema (written by
``scripts/evaluate_precrime.py``)::

    {
      "generated_at": "<ISO-8601 timestamp>",
      "n_videos": <int>,
      "fps": <float>,
      "threshold": <float>,
      "malt": {
        "malt_frames": <float>,
        "malt_seconds": <float>,
        "per_video_frames": [<float>, ...]
      },
      "ear_curve": {
        "thresholds": [<float>, ...],
        "ear_values": [<float>, ...]
      },
      "temporal_roc": {
        "<L>": {
          "fpr": [<float>, ...],
          "tpr": [<float>, ...],
          "auc": <float>
        },
        ...
      }
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, list[float]]


@dataclass
class MALTResult:
    """Mean Alert Lead Time result.

    Attributes
    ----------
    malt_frames:
        Mean lead time in frames averaged over all crime videos.  MALT = 0
        when every alarm fires exactly at onset (or after); MALT > 0 when
        alarms fire before onset.
    malt_seconds:
        ``malt_frames / fps``.
    per_video_frames:
        Per-video lead time in frames.  0 for videos with no pre-crime alarm.
    """

    malt_frames: float
    malt_seconds: float
    per_video_frames: list[float] = field(default_factory=list)


@dataclass
class EARCurveResult:
    """Early Alert Rate curve over a range of thresholds.

    Attributes
    ----------
    thresholds:
        Ascending threshold values at which EAR was computed.
    ear_values:
        EAR at each threshold.  EAR is the fraction of crime videos where at
        least one pre-crime frame (before crime onset) has score ≥ threshold.
        Monotonically non-increasing as threshold increases.
    """

    thresholds: np.ndarray
    ear_values: np.ndarray


@dataclass
class ROCResult:
    """ROC curve and AUC for a single lead-time condition.

    Attributes
    ----------
    fpr:    False positive rates.
    tpr:    True positive rates (corresponding to ``fpr``).
    auc:    Area under the ROC curve (T-AUC@L).
    """

    fpr: np.ndarray
    tpr: np.ndarray
    auc: float


# ---------------------------------------------------------------------------
# Helper: ROC from per-video scores
# ---------------------------------------------------------------------------


def _roc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> ROCResult:
    """Compute ROC curve and AUC from per-video summary scores."""
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return ROCResult(
            fpr=np.array([0.0, 1.0]),
            tpr=np.array([0.0, 1.0]),
            auc=0.5,
        )

    # Compute AUC via pairwise ranking.
    concordant = float(np.sum(pos_scores[:, None] > neg_scores[None, :])) + 0.5 * float(
        np.sum(pos_scores[:, None] == neg_scores[None, :])
    )
    auc = concordant / (len(pos_scores) * len(neg_scores))

    # Build ROC curve: scan over all unique thresholds in descending order.
    thresholds = np.sort(np.unique(np.concatenate([pos_scores, neg_scores])))[::-1]
    fprs = []
    tprs = []
    for t in thresholds:
        tprs.append(float(np.mean(pos_scores >= t)))
        fprs.append(float(np.mean(neg_scores >= t)))
    fprs.append(0.0)
    tprs.append(0.0)
    return ROCResult(
        fpr=np.array([1.0] + fprs, dtype=float),
        tpr=np.array([1.0] + tprs, dtype=float),
        auc=float(auc),
    )


# ---------------------------------------------------------------------------
# PreCrimeMetrics
# ---------------------------------------------------------------------------


class PreCrimeMetrics:
    """Compute pre-crime lead-time metrics over a collection of videos.

    All methods accept either ``np.ndarray`` or ``list[float]`` for score
    arrays, satisfying the acceptance criterion "All metrics accept ndarray or
    list[float] input."

    Parameters
    ----------
    default_fps:
        Default frame rate used when ``fps`` is not supplied to methods.
    """

    def __init__(self, default_fps: float = 30.0) -> None:
        self.default_fps = default_fps

    # ------------------------------------------------------------------
    # 1. Mean Alert Lead Time
    # ------------------------------------------------------------------

    def mean_alert_lead_time(
        self,
        scores: list[ArrayLike],
        crime_starts: list[int],
        fps: float | None = None,
        threshold: float = 0.5,
    ) -> MALTResult:
        """Compute Mean Alert Lead Time (MALT) over a set of crime videos.

        For each video the lead time is defined as::

            lead_frames = onset_frame - first_pre_crime_alarm_frame

        where *first_pre_crime_alarm_frame* is the earliest frame *before*
        ``onset_frame`` with score ≥ ``threshold``.  If no such frame exists
        the per-video lead time is 0.

        MALT is the mean lead time across all videos.

        Parameters
        ----------
        scores:
            Per-video score arrays.  Each array has shape ``(T_v,)`` and
            contains threat scores in [0, 1].
        crime_starts:
            Crime onset frame index for each video.
        fps:
            Frame rate; defaults to ``self.default_fps``.
        threshold:
            Alert decision threshold.

        Returns
        -------
        MALTResult
        """
        if fps is None:
            fps = self.default_fps

        per_video: list[float] = []
        for video_scores, onset in zip(scores, crime_starts):
            arr = np.asarray(video_scores, dtype=float)
            if onset <= 0:
                per_video.append(0.0)
                continue
            pre_crime = arr[:onset]
            alarm_indices = np.where(pre_crime >= threshold)[0]
            if len(alarm_indices) == 0:
                per_video.append(0.0)
            else:
                per_video.append(float(onset - int(alarm_indices[0])))

        malt_frames = float(np.mean(per_video)) if per_video else 0.0
        return MALTResult(
            malt_frames=malt_frames,
            malt_seconds=malt_frames / fps,
            per_video_frames=per_video,
        )

    # ------------------------------------------------------------------
    # 2. EAR Curve
    # ------------------------------------------------------------------

    def ear_curve(
        self,
        scores: list[ArrayLike],
        crime_starts: list[int],
        thresholds: ArrayLike | None = None,
    ) -> EARCurveResult:
        """Compute the Early Alert Rate (EAR) curve over a range of thresholds.

        EAR at threshold *t* = fraction of crime videos where at least one
        pre-crime frame has score ≥ *t*.  The curve is guaranteed to be
        monotonically non-increasing.

        Parameters
        ----------
        scores:
            Per-video score arrays.
        crime_starts:
            Crime onset frame index for each video.
        thresholds:
            Thresholds to sweep.  Defaults to np.arange(0.1, 0.95, 0.05) —
            seventeen evenly spaced values covering [0.1, 0.9].

        Returns
        -------
        EARCurveResult
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.91, 0.05)
        thresholds_arr = np.sort(np.asarray(thresholds, dtype=float))

        # Pre-compute per-video max pre-crime score.
        max_pre_scores: list[float] = []
        for video_scores, onset in zip(scores, crime_starts):
            arr = np.asarray(video_scores, dtype=float)
            if onset <= 0:
                max_pre_scores.append(-np.inf)
            else:
                max_pre_scores.append(float(np.max(arr[:onset])))
        max_pre_arr = np.array(max_pre_scores)

        n = len(max_pre_scores)
        if n == 0:
            return EARCurveResult(
                thresholds=thresholds_arr,
                ear_values=np.zeros_like(thresholds_arr),
            )

        ear_values = np.array(
            [float(np.mean(max_pre_arr >= t)) for t in thresholds_arr]
        )
        return EARCurveResult(thresholds=thresholds_arr, ear_values=ear_values)

    # ------------------------------------------------------------------
    # 3. Temporal ROC
    # ------------------------------------------------------------------

    def temporal_roc(
        self,
        scores: list[ArrayLike],
        crime_starts: list[int | None],
        lead_times: list[int] | None = None,
    ) -> dict[int, ROCResult]:
        """Compute Temporal ROC curves conditioned on detection lead-time ≥ L.

        For each lead-time *L*:

        * Crime videos (``crime_starts[v]`` is an ``int``): the summary score
          is ``max(scores[v][:onset - L])`` when ``onset > L``, else 0.0.
          These videos are labelled positive (y_true = 1).
        * Non-crime videos (``crime_starts[v]`` is ``None``): the summary
          score is ``max(scores[v])``.
          These videos are labelled negative (y_true = 0).

        T-AUC@L is the area under the resulting ROC and is the primary result
        table metric for the paper.  T-AUC@0 ≥ T-AUC@30 ≥ T-AUC@60 is the
        expected ordering but is not enforced.

        Parameters
        ----------
        scores:
            Per-video score arrays.
        crime_starts:
            Crime onset frame per video, or ``None`` for non-crime videos.
        lead_times:
            Lead-time values for conditioning; defaults to ``[0, 30, 60, 90]``.

        Returns
        -------
        dict mapping each lead-time *L* to a :class:`ROCResult`.
        """
        if lead_times is None:
            lead_times = [0, 30, 60, 90]

        results: dict[int, ROCResult] = {}
        for L in lead_times:
            y_true_list: list[int] = []
            y_score_list: list[float] = []
            for video_scores, onset in zip(scores, crime_starts):
                arr = np.asarray(video_scores, dtype=float)
                if onset is None:
                    # Non-crime video — negative example.
                    y_true_list.append(0)
                    y_score_list.append(float(np.max(arr)) if len(arr) > 0 else 0.0)
                else:
                    # Crime video — positive example.
                    y_true_list.append(1)
                    cutoff = max(0, int(onset) - L)
                    if cutoff == 0:
                        y_score_list.append(0.0)
                    else:
                        y_score_list.append(float(np.max(arr[:cutoff])))

            y_true_arr = np.array(y_true_list, dtype=np.int32)
            y_score_arr = np.array(y_score_list, dtype=float)
            results[L] = _roc_from_scores(y_true_arr, y_score_arr)

        return results
