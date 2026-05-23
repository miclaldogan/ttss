"""Temporal Threat Scoring System (TTSS): training orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ttss.models.ttss_pipeline import TtssPipeline
from ttss.training.losses import composite_threat_loss


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for TTSS training loops."""

    epochs: int = 20
    learning_rate: float = 1e-4
    batch_size: int = 8


@dataclass(slots=True)
class TrainResult:
    """Summary of a training run."""

    train_loss: float = 0.0
    val_loss: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


class Trainer:
    """Trainer skeleton for TTSS experiments."""

    def __init__(
        self,
        pipeline: TtssPipeline,
        config: TrainerConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or TrainerConfig()

    def fit(
        self,
        train_batches: Sequence[Sequence[Any]],
        train_targets: Sequence[float],
        val_batches: Sequence[Sequence[Any]] | None = None,
        val_targets: Sequence[float] | None = None,
    ) -> TrainResult:
        """Run a minimal training loop scaffold."""
        train_scores = [self.pipeline.predict_from_frames(batch).score for batch in train_batches]
        train_loss = composite_threat_loss(train_scores, train_targets)

        val_loss = 0.0
        if val_batches is not None and val_targets is not None:
            val_scores = [
                self.pipeline.predict_from_frames(batch).score for batch in val_batches
            ]
            val_loss = composite_threat_loss(val_scores, val_targets)

        return TrainResult(
            train_loss=train_loss,
            val_loss=val_loss,
            metrics={"train_loss": train_loss, "val_loss": val_loss},
        )
