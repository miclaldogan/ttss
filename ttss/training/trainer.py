"""Temporal Threat Scoring System (TTSS): training orchestration."""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ttss.models.ttss_pipeline import TtssPipeline
from ttss.training.losses import (
    TemporalConsistencyLoss,
    ThreatScoreRegressionLoss,
    composite_threat_loss,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for TTSS training loops."""

    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 4
    max_grad_norm: float = 1.0
    lambda_reg: float = 1.0         # regression loss weight
    lambda_tc: float = 0.1          # temporal consistency loss weight
    lambda_pre: float = 0.5         # pre-crime detection loss weight
    # backward-compat aliases kept so existing call sites don't break
    lambda1: float = 1.0
    lambda2: float = 0.1
    vit_lr_scale: float = 0.1       # ViT fine-tune LR = learning_rate * vit_lr_scale
    warmup_steps: int = 0           # linear warmup steps (0 = no warmup)
    patience: int = 5               # early stopping patience on val AUC
    use_wandb: bool = False
    wandb_project: str = "ttss"
    checkpoint_dir: str = "checkpoints"
    mixed_precision: bool = True    # auto-disabled when not on CUDA
    use_tensorboard: bool = False
    tensorboard_dir: str = "runs/ttss"
    dry_run: bool = False
    dry_run_steps: int = 2


@dataclass(slots=True)
class TrainResult:
    """Summary of a training run."""

    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val_auc: float = 0.0
    epochs_completed: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backward-compat skeleton (kept for existing tests)
# ---------------------------------------------------------------------------


class Trainer:
    """Trainer skeleton for TTSS experiments (backward-compat facade)."""

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


# ---------------------------------------------------------------------------
# Production trainer
# ---------------------------------------------------------------------------


class TTSSTrainer:
    """Full-featured TTSS trainer with gradient clipping, LR scheduling,
    mixed precision, checkpointing, and optional W&B logging.

    Architecture::

        optimizer  : AdamW
        scheduler  : CosineAnnealingLR (T_max = epochs)
        loss       : λ1 * ThreatScoreRegressionLoss + λ2 * TemporalConsistencyLoss
        grad_clip  : max_norm = config.max_grad_norm
        amp        : torch.amp.autocast (CPU-safe; only uses bfloat16 when CUDA)
    """

    def __init__(self, model: nn.Module, config: TrainerConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        self._best_val_auc: float = 0.0
        self._patience_counter: int = 0
        self._wandb_run: Any = None

        # Differential LR: ViT fine-tune blocks use vit_lr_scale × base LR.
        # Detected by checking whether the model has a .vit attribute with
        # trainable parameters (i.e. EndToEndThreatModel usage).
        vit_trainable = []
        vit_module = getattr(model, "vit", None)
        if vit_module is not None:
            vit_trainable = [p for p in vit_module.parameters() if p.requires_grad]

        if vit_trainable:
            vit_param_ids = {id(p) for p in vit_trainable}
            bilstm_params = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in vit_param_ids
            ]
            param_groups: Any = [
                {"params": bilstm_params, "lr": self.config.learning_rate},
                {"params": vit_trainable, "lr": self.config.learning_rate * self.config.vit_lr_scale},
            ]
            _logger.info(
                "Differential LR — BiLSTM: %.2e  ViT fine-tune (%d params): %.2e",
                self.config.learning_rate,
                sum(p.numel() for p in vit_trainable),
                self.config.learning_rate * self.config.vit_lr_scale,
            )
        else:
            param_groups = model.parameters()

        self.optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, self.config.epochs))
        self._global_step: int = 0
        self._tb_writer: Any = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=self.config.tensorboard_dir)
                _logger.info("TensorBoard writer initialised at %s", self.config.tensorboard_dir)
            except ImportError:
                _logger.warning("tensorboard not installed — TB logging disabled")
        self._regression_loss = ThreatScoreRegressionLoss()
        self._consistency_loss = TemporalConsistencyLoss()

        device = next(model.parameters()).device
        self._use_amp = self.config.mixed_precision and device.type == "cuda"
        self._scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler() if self._use_amp else None
        )

        _logger.info(
            "TTSSTrainer initialised: epochs=%d lr=%g lambda1=%g lambda2=%g amp=%s",
            self.config.epochs,
            self.config.learning_rate,
            self.config.lambda1,
            self.config.lambda2,
            self._use_amp,
        )

    # ------------------------------------------------------------------
    # W&B helpers
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        if not self.config.use_wandb:
            return
        if os.environ.get("WANDB_MODE") == "disabled":
            _logger.info("WANDB_MODE=disabled — skipping W&B initialisation")
            return
        try:
            import wandb  # type: ignore[import]
            self._wandb_run = wandb.init(project=self.config.wandb_project)
            _logger.info("W&B run initialised: %s", self._wandb_run.name)
        except ImportError:
            _logger.warning("wandb not installed — logging disabled")

    def _log_wandb(self, metrics: dict[str, float], step: int) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Run one forward + backward + optimiser step.

        Args:
            x: Input feature tensor ``(B, T, F)``.
            y: Target threat scores ``(B, T)`` in ``[0, 1]``.

        Returns:
            Scalar loss value as a Python float.
        """
        self.model.train()
        self.optimizer.zero_grad()

        device = next(self.model.parameters()).device
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device) for t in x)
        else:
            x = x.to(device)
        y = y.to(device)

        amp_ctx: Any = (
            torch.amp.autocast(device_type="cuda")
            if self._use_amp
            else torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with amp_ctx:
            result = self.model(x)
            # BiLSTMThreatPredictor returns ThreatPrediction with frame_scores (B,T)
            preds = result.frame_scores if hasattr(result, "frame_scores") else result
            reg = self._regression_loss(preds, y)
            cons = self._consistency_loss(preds)
            loss = self.config.lambda1 * reg + self.config.lambda2 * cons

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        step_loss = float(loss.detach().cpu().item())
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("train/loss_step", step_loss, self._global_step)
        self._global_step += 1
        return step_loss

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | pathlib.Path,
        epoch: int,
        val_auc: float = 0.0,
    ) -> None:
        """Save model state dict + training state to *path*."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "val_auc": val_auc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        _logger.info("Checkpoint saved: %s (epoch=%d val_auc=%.4f)", path, epoch, val_auc)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | pathlib.Path,
        model: nn.Module,
        config: TrainerConfig | None = None,
    ) -> "TTSSTrainer":
        """Restore a trainer from a checkpoint file.

        Returns a new :class:`TTSSTrainer` with the model and optimiser state
        loaded from *path*.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        trainer = cls(model, config)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        trainer._best_val_auc = float(ckpt.get("val_auc", 0.0))
        _logger.info(
            "Checkpoint loaded from %s (epoch=%d val_auc=%.4f)",
            path,
            ckpt.get("epoch", -1),
            trainer._best_val_auc,
        )
        return trainer

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
        val_iter: Iterator[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> TrainResult:
        """Run the full training loop.

        Args:
            train_iter: Yields ``(x, y)`` tensor pairs per epoch.
            val_iter:   Optional validation yields; used for AUC tracking.

        Returns:
            :class:`TrainResult` with final losses and best val AUC.
        """
        self._init_wandb()
        checkpoint_dir = pathlib.Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs = self.config.dry_run_steps if self.config.dry_run else self.config.epochs
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_steps = 0
            for x, y in train_iter:
                step_loss = self.train_step(x, y)
                epoch_loss += step_loss
                n_steps += 1
                if self.config.dry_run and n_steps >= self.config.dry_run_steps:
                    break

            avg_loss = epoch_loss / max(1, n_steps)
            epoch_losses.append(avg_loss)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validation pass — compute loss, AUC, EAR
            val_loss, val_auc, val_ear = 0.0, 0.0, 0.0
            if val_iter is not None:
                import numpy as np
                from ttss.training.metrics import frame_level_auc, early_alert_rate
                val_device = next(self.model.parameters()).device
                self.model.eval()
                n_val = 0
                all_preds: list[float] = []
                all_targets: list[float] = []
                with torch.no_grad():
                    for x_val, y_val in val_iter:
                        if isinstance(x_val, (tuple, list)):
                            x_val = tuple(t.to(val_device) for t in x_val)
                        else:
                            x_val = x_val.to(val_device)
                        y_val = y_val.to(val_device)
                        result = self.model(x_val)
                        preds = result.frame_scores if hasattr(result, "frame_scores") else result
                        reg = self._regression_loss(preds, y_val)
                        cons = self._consistency_loss(preds)
                        val_loss += float((self.config.lambda1 * reg + self.config.lambda2 * cons).item())
                        all_preds.extend(preds.detach().cpu().reshape(-1).tolist())
                        all_targets.extend(y_val.detach().cpu().reshape(-1).tolist())
                        n_val += 1
                val_loss = val_loss / max(1, n_val)
                if all_preds:
                    y_true_np = np.array(all_targets)
                    y_score_np = np.array(all_preds)
                    # binarise targets at 0.5 for AUC/EAR
                    y_bin = (y_true_np >= 0.5).astype(int)
                    if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                        val_auc = frame_level_auc(y_bin, y_score_np)
                        val_ear = early_alert_rate(y_bin, y_score_np)
                self.model.train()

                # Early stopping
                if val_auc > self._best_val_auc:
                    self._best_val_auc = val_auc
                    self._patience_counter = 0
                    self.save_checkpoint(checkpoint_dir / "best.pt", epoch=epoch, val_auc=val_auc)
                else:
                    self._patience_counter += 1
                    if not self.config.dry_run and self._patience_counter >= self.config.patience:
                        _logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.config.patience)
                        break

            metrics = {"train_loss": avg_loss, "val_loss": val_loss, "val_auc": val_auc,
                       "val_ear": val_ear, "lr": current_lr, "epoch": epoch}
            self._log_wandb(metrics, step=epoch)
            _logger.debug(
                "epoch=%d train_loss=%.4f val_loss=%.4f val_auc=%.4f val_ear=%.4f lr=%.2e",
                epoch, avg_loss, val_loss, val_auc, val_ear, current_lr,
            )

            if self._tb_writer is not None:
                self._tb_writer.add_scalar("train/loss_epoch", avg_loss, epoch)
                self._tb_writer.add_scalar("train/lr", current_lr, epoch)
                if val_iter is not None:
                    self._tb_writer.add_scalar("val/loss", val_loss, epoch)
                    self._tb_writer.add_scalar("val/auc", val_auc, epoch)
                    self._tb_writer.add_scalar("val/ear", val_ear, epoch)

            # Always save latest checkpoint
            self.save_checkpoint(checkpoint_dir / "latest.pt", epoch=epoch, val_auc=self._best_val_auc)

            if self.config.dry_run:
                break

        train_loss = epoch_losses[-1] if epoch_losses else 0.0
        if self._wandb_run is not None:
            self._wandb_run.finish()
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()

        return TrainResult(
            train_loss=train_loss,
            val_loss=val_loss if val_iter is not None else 0.0,
            best_val_auc=self._best_val_auc,
            epochs_completed=len(epoch_losses),
            metrics={"train_loss": train_loss, "val_loss": val_loss if val_iter is not None else 0.0},
        )

