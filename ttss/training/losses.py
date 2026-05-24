"""Temporal Threat Scoring System (TTSS): temporal-aware loss functions.

These losses support continuous threat scoring with explicit smoothness and
pre-crime emphasis terms.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class TemporalConsistencyLoss(nn.Module):
    """Penalize abrupt changes between consecutive frame scores.

    Only differences **greater than** *delta* are penalised, giving the model
    tolerance for small frame-to-frame fluctuations.
    """

    def __init__(self, delta: float = 0.0, reduction: str = "mean", p: int = 1) -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.p = p

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        if scores.size(-1) < 2:
            return scores.new_tensor(0.0)
        deltas = torch.diff(scores, dim=-1).abs()
        if self.delta > 0.0:
            deltas = torch.clamp(deltas - self.delta, min=0.0)
        if self.p == 2:
            deltas = deltas.pow(2)
        if self.reduction == "sum":
            return deltas.sum()
        if self.reduction == "none":
            return deltas
        return deltas.mean()


class ThreatScoreRegressionLoss(nn.Module):
    """Weighted MSE loss for continuous TTSS threat score regression."""

    def __init__(self, base_weight: float = 1.0, temporal_weight: float = 1.0) -> None:
        super().__init__()
        self.base_weight = base_weight
        self.temporal_weight = temporal_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        temporal_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predictions = predictions.float()
        targets = targets.float()
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have the same shape")

        weights = (
            temporal_weights.float()
            if temporal_weights is not None
            else self.base_weight + (self.temporal_weight * targets)
        )
        return torch.mean(weights * (predictions - targets).pow(2))


class PreCrimeDetectionLoss(nn.Module):
    """Extra weighted regression penalty for the pre-crime window."""

    def __init__(self, precrime_weight: float = 2.0) -> None:
        super().__init__()
        self.precrime_weight = precrime_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        precrime_mask: torch.Tensor,
    ) -> torch.Tensor:
        predictions = predictions.float()
        targets = targets.float()
        mask = precrime_mask.bool()
        if predictions.shape != targets.shape or predictions.shape != mask.shape:
            raise ValueError("predictions, targets, and precrime_mask must share a shape")
        if not torch.any(mask):
            return predictions.new_tensor(0.0)
        squared_error = (predictions - targets).pow(2)
        weights = torch.where(
            mask,
            torch.full_like(squared_error, self.precrime_weight),
            torch.ones_like(squared_error),
        )
        masked_error = squared_error * weights * mask.float()
        return masked_error.sum() / mask.float().sum()


def mse_loss(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Compute mean squared error for scalar threat scores."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        return 0.0
    squared_errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    return sum(squared_errors) / len(squared_errors)


def temporal_consistency_loss(scores: Sequence[float]) -> float:
    """Penalize abrupt score changes across adjacent timesteps."""
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    return float(TemporalConsistencyLoss()(score_tensor).item())


def composite_threat_loss(
    predictions: Sequence[float],
    targets: Sequence[float],
    lambda1: float = 1.0,
    lambda2: float = 0.1,
    consistency_weight: float | None = None,
) -> float:
    """Combine regression loss with a temporal smoothness prior.

    ``L = λ1 * regression + λ2 * consistency``

    Args:
        predictions:       Per-frame predicted threat scores.
        targets:           Per-frame ground-truth threat scores.
        lambda1:           Weight for the regression term (default 1.0).
        lambda2:           Weight for the consistency term (default 0.1).
        consistency_weight: Deprecated alias for *lambda2*; takes precedence
                            when provided for backwards compatibility.
    """
    lam2 = consistency_weight if consistency_weight is not None else lambda2
    if isinstance(predictions, torch.Tensor):
        prediction_tensor = predictions.detach().clone().float()
    else:
        prediction_tensor = torch.tensor(predictions, dtype=torch.float32)
    if isinstance(targets, torch.Tensor):
        target_tensor = targets.detach().clone().float()
    else:
        target_tensor = torch.tensor(targets, dtype=torch.float32)
    regression = ThreatScoreRegressionLoss()(prediction_tensor, target_tensor)
    smoothness = TemporalConsistencyLoss()(prediction_tensor)
    return float((lambda1 * regression + lam2 * smoothness).item())
