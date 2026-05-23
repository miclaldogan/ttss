"""Temporal Threat Scoring System (TTSS): BiLSTM threat predictor.

This module implements the temporal prediction layer used by TTSS. It consumes
fused recognition and scene features and produces frame-wise threat scores in
the range [0, 1] using a bidirectional LSTM and temporal attention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ThreatPrediction:
    """Prediction result from the temporal threat head."""

    frame_scores: torch.Tensor
    sequence_score: torch.Tensor
    attention_weights: torch.Tensor
    hidden_state: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    @property
    def score(self) -> float:
        """Return a Python float summary score for compatibility."""

        return float(self.sequence_score.detach().cpu().reshape(-1)[0].item())


class TemporalAttention(nn.Module):
    """Temporal attention block over BiLSTM hidden states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention weights and the aggregated temporal context."""
        logits = self.score(torch.tanh(self.proj(hidden_states))).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return weights, context


class BiLSTMThreatPredictor(nn.Module):
    """BiLSTM threat predictor with temporal attention and sigmoid outputs."""

    def __init__(
        self,
        input_dim: int = 770,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim

        self.input_projection = nn.Linear(input_dim, projection_dim)
        self.projection_norm = nn.LayerNorm(projection_dim)
        self.lstm = nn.LSTM(
            # two layers of BiLSTM: hidden_dim*2 output at each time step
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_activation = nn.Sigmoid()

        n_params = sum(p.numel() for p in self.parameters())
        _logger.info(
            "BiLSTMThreatPredictor initialised: input_dim=%d hidden_dim=%d "
            "num_layers=%d params=%d",
            input_dim, hidden_dim, num_layers, n_params,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> ThreatPrediction | tuple[torch.Tensor, torch.Tensor]:
        """Predict per-frame threat scores from a fused feature tensor.

        Args:
            sequence_features: Tensor of shape ``(B, T, F)`` or ``(T, F)``
                where F is the fused feature dimension (e.g. 1536 for
                YOLO-8 + ViT-768 × 2).
            mask: Optional boolean tensor of shape ``(B, T)`` marking valid
                time steps (True = keep).
            return_attention: When ``True``, return a ``(frame_scores, attn)``
                tuple instead of a :class:`ThreatPrediction` object, where
                ``frame_scores`` has shape ``(B, T)`` and ``attn`` has shape
                ``(B, T)``.  Useful for inspection and visualisation.

        Returns:
            :class:`ThreatPrediction` (default) or
            ``tuple[Tensor[B, T], Tensor[B, T]]`` when *return_attention* is
            ``True``.
        """
        if sequence_features.ndim == 2:
            sequence_features = sequence_features.unsqueeze(0)
        if sequence_features.ndim != 3:
            raise ValueError("sequence_features must have shape (B, T, F) or (T, F)")

        projected = self.projection_norm(self.input_projection(sequence_features))
        hidden_states, _ = self.lstm(projected)
        attention_weights, context = self.attention(hidden_states, mask=mask)
        expanded_context = context.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        logits = self.output_projection(torch.cat([hidden_states, expanded_context], dim=-1))
        frame_scores = self.output_activation(logits).squeeze(-1)
        sequence_score = torch.sum(frame_scores * attention_weights, dim=-1)

        if return_attention:
            return frame_scores, attention_weights

        return ThreatPrediction(
            frame_scores=frame_scores,
            sequence_score=sequence_score,
            attention_weights=attention_weights,
            hidden_state=hidden_states,
        )

    def predict_sequence(
        self,
        sequence_features: Sequence[Sequence[float]] | torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> ThreatPrediction:
        """Convenience wrapper for list-based inference inputs."""
        if isinstance(sequence_features, torch.Tensor):
            tensor = sequence_features.to(dtype=torch.float32)
        else:
            if not sequence_features:
                empty = torch.empty((1, 0), dtype=torch.float32)
                return ThreatPrediction(
                    frame_scores=empty,
                    sequence_score=torch.zeros(1, dtype=torch.float32),
                    attention_weights=empty,
                    hidden_state=torch.empty((1, 0, self.hidden_dim * 2)),
                )
            tensor = torch.tensor(sequence_features, dtype=torch.float32)
        return self.forward(tensor, mask=mask)


BiLstmThreatPredictor = BiLSTMThreatPredictor
