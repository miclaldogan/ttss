"""Temporal Threat Scoring System (TTSS): statistical evaluation utilities.

Three tools for publication-quality results:

1. bootstrap_ci       — 95% confidence intervals for any scalar metric via
                         non-parametric bootstrap resampling.
2. permutation_test   — p-value for H0: model A is no better than model B,
                         via label-permutation significance test.
3. PlattCalibrator    — sigmoid score calibration (Platt scaling) so that
                         output scores match empirical probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Output of :func:`bootstrap_ci`."""

    mean: float
    lower: float
    upper: float
    ci: float
    n_bootstrap: int

    def __str__(self) -> str:
        pct = int(self.ci * 100)
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({pct}% CI, n={self.n_bootstrap})"


@dataclass
class PermutationResult:
    """Output of :func:`permutation_test`."""

    observed_diff: float
    p_value: float
    n_permutations: int
    significant: bool

    def __str__(self) -> str:
        sig = "✓" if self.significant else "✗"
        return (
            f"Δ={self.observed_diff:+.4f}  p={self.p_value:.4f}  "
            f"significant={sig}  (n_perm={self.n_permutations})"
        )


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Estimate a confidence interval for *metric_fn* via bootstrap resampling.

    Parameters
    ----------
    y_true:      Ground-truth binary labels.
    y_score:     Predicted scores.
    metric_fn:   ``(y_true, y_score) -> float`` — e.g. ``frame_level_auc``.
    n_bootstrap: Number of bootstrap resamples (default 1 000).
    ci:          Confidence level, e.g. 0.95 for a 95% CI.
    seed:        Random seed for reproducibility.

    Returns
    -------
    :class:`BootstrapResult` with mean, lower, and upper bounds.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)

    boot_scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores.append(metric_fn(y_true[idx], y_score[idx]))

    boot_arr = np.array(boot_scores)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_arr, alpha))
    upper = float(np.quantile(boot_arr, 1.0 - alpha))
    return BootstrapResult(
        mean=float(np.mean(boot_arr)),
        lower=lower,
        upper=upper,
        ci=ci,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Permutation significance test
# ---------------------------------------------------------------------------


def permutation_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PermutationResult:
    """Test whether model A is significantly better than model B.

    H0: the two score arrays come from the same distribution (no difference).
    The test statistic is ``metric(A) - metric(B)``.  Under H0 we permute
    which score array is labelled A vs B for each sample and recompute the
    statistic, building a null distribution.

    Parameters
    ----------
    y_true:         Ground-truth binary labels shared by both models.
    scores_a:       Predicted scores from model A.
    scores_b:       Predicted scores from model B.
    metric_fn:      ``(y_true, y_score) -> float``.
    n_permutations: Number of permutation resamples.
    alpha:          Significance level (default 0.05).
    seed:           Random seed.

    Returns
    -------
    :class:`PermutationResult` with p-value and significance flag.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    observed_diff = metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b)

    null_diffs: list[float] = []
    for _ in range(n_permutations):
        mask = rng.integers(0, 2, size=len(y_true)).astype(bool)
        perm_a = np.where(mask, scores_a, scores_b)
        perm_b = np.where(mask, scores_b, scores_a)
        null_diffs.append(metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b))

    null_arr = np.array(null_diffs)
    p_value = float(np.mean(null_arr >= observed_diff))
    return PermutationResult(
        observed_diff=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        significant=p_value < alpha,
    )


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """Sigmoid (Platt) score calibration.

    Fits a logistic regression ``P(y=1|s) = σ(A·s + B)`` on a held-out
    calibration set so that output scores approximate true probabilities.

    Usage::

        cal = PlattCalibrator()
        cal.fit(val_scores, val_labels)
        calibrated = cal.transform(test_scores)
    """

    def __init__(self) -> None:
        self._A: float = 1.0
        self._B: float = 0.0
        self._fitted: bool = False

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> "PlattCalibrator":
        """Fit sigmoid parameters A, B via gradient descent on log-loss.

        Parameters
        ----------
        scores:   Raw model scores in any range.
        labels:   Binary ground-truth labels (0 or 1).
        max_iter: Gradient descent steps.
        lr:       Learning rate.
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        A, B = 1.0, 0.0
        for _ in range(max_iter):
            logits = A * scores + B
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            err = probs - labels
            dA = float(np.mean(err * scores))
            dB = float(np.mean(err))
            A -= lr * dA
            B -= lr * dB
        self._A, self._B = A, B
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in [0, 1]."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        scores = np.asarray(scores, dtype=float)
        logits = self._A * scores + self._B
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and return calibrated scores in one call."""
        return self.fit(scores, labels).transform(scores)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


@dataclass
class WilcoxonResult:
    """Output of :func:`wilcoxon_test`."""

    statistic: float
    p_value: float
    significant: bool

    def __str__(self) -> str:
        sig = "✓" if self.significant else "✗"
        return f"W={self.statistic:.2f}  p={self.p_value:.4f}  significant={sig}"


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Bonferroni correction for multiple comparisons.

    Adjusts the significance threshold by dividing *alpha* by the number of
    tests.  Each p-value is flagged as significant if ``p < alpha / n_tests``.

    Parameters
    ----------
    p_values: Raw p-values from ``n_tests`` independent tests.
    alpha:    Family-wise error rate (default 0.05).

    Returns
    -------
    List of booleans; ``True`` means the test is significant after correction.
    """
    n = len(p_values)
    if n == 0:
        return []
    threshold = alpha / n
    return [p < threshold for p in p_values]


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test: H0 — A and B come from the same distribution.

    Use on paired per-video metric values (e.g. per-video AUC from two models).
    Reports a p-value with Bonferroni correction when called for multiple baselines.

    Parameters
    ----------
    scores_a, scores_b: Paired per-video metric values (equal length).
    alpha:              Significance level (default 0.05).
    alternative:        ``'two-sided'``, ``'greater'``, or ``'less'``.
    """
    try:
        from scipy.stats import wilcoxon as _wilcoxon
    except ImportError as exc:
        raise ImportError("scipy is required for wilcoxon_test — pip install scipy") from exc

    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if np.all(scores_a == scores_b):
        return WilcoxonResult(statistic=0.0, p_value=1.0, significant=False)
    result = _wilcoxon(scores_a, scores_b, alternative=alternative)
    return WilcoxonResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=float(result.pvalue) < alpha,
    )
