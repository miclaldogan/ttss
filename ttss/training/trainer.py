"""Temporal Threat Scoring System (TTSS): training orchestration."""

from __future__ import annotations

import logging
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ttss.models.ttss_pipeline import TtssPipeline
from ttss.training.losses import (
    MILRankingLoss,
    PreCrimeDetectionLoss,
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
    steps_per_epoch: int = 0   # 0 = use full dataset; set to len(ds)//batch_size
    val_steps: int = 0          # 0 = unlimited (infinite loop bug!); set to len(val_ds)//batch_size
    grad_accum_steps: int = 1   # gradient accumulation; effective batch = batch_size * grad_accum_steps

    # Loss weights
    lambda_reg: float = 1.0         # regression loss weight
    lambda_tc: float = 0.1          # temporal consistency loss weight
    lambda_pre: float = 0.5         # pre-crime detection loss weight
    lambda_mil: float = 1.0         # MIL ranking loss weight
    use_mil_loss: bool = True        # MIL ranking loss (requires anomaly+normal in batch)

    # Backward-compat aliases — stay in sync with lambda_reg/lambda_tc
    lambda1: float = 1.0
    lambda2: float = 0.1

    vit_lr_scale: float = 0.1       # ViT fine-tune LR = learning_rate * vit_lr_scale
    yolo_lr_scale: float = 1.0      # YOLO/flow branch LR = learning_rate * yolo_lr_scale (use >1 when fine-tuning on enriched data)
    warmup_steps: int = 100         # linear warmup steps (0 = no warmup)
    patience: int = 5               # early stopping patience on val AUC

    use_wandb: bool = False
    wandb_project: str = "ttss"
    checkpoint_dir: str = "checkpoints"
    mixed_precision: bool = True
    use_tensorboard: bool = False
    tensorboard_dir: str = "runs/ttss"
    dry_run: bool = False
    dry_run_steps: int = 2
    val_annotations: str = ""   # path to test_annotations.json for frame-level GT


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
    """Full-featured TTSS trainer.

    Loss::

        L = λ_reg * Regression
          + λ_tc  * TemporalConsistency
          + λ_pre * PreCrimeDetection   (when precrime_mask provided)
          + λ_mil * MILRanking          (when batch has anomaly + normal clips)

    Scheduler::

        CosineWarmupScheduler  (linear warmup then cosine decay)
        — falls back to CosineAnnealingLR when warmup_steps == 0

    AMP::
        torch.amp.autocast on CUDA; no-op on CPU
    """

    def __init__(self, model: nn.Module, config: TrainerConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        self._best_val_auc: float = 0.0
        self._patience_counter: int = 0
        self._wandb_run: Any = None

        # Differential LR for EndToEndThreatModel (ViT blocks vs BiLSTM)
        vit_trainable = []
        vit_module = getattr(model, "vit", None)
        if vit_module is not None:
            vit_trainable = [p for p in vit_module.parameters() if p.requires_grad]

        # Detect YOLO/flow branch parameters for differential LR
        yolo_modules = []
        for name in ("yolo_gate", "yolo_residual"):
            m = getattr(model, name, None)
            if m is not None:
                yolo_modules.append(m)
        yolo_param_ids = {id(p) for m in yolo_modules for p in m.parameters()}

        if vit_trainable:
            vit_param_ids = {id(p) for p in vit_trainable}
            bilstm_params = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in vit_param_ids and id(p) not in yolo_param_ids
            ]
            param_groups_list: list[dict] = [
                {"params": bilstm_params, "lr": self.config.learning_rate},
                {"params": vit_trainable,  "lr": self.config.learning_rate * self.config.vit_lr_scale},
            ]
            if yolo_param_ids:
                yolo_params = [p for p in model.parameters() if p.requires_grad and id(p) in yolo_param_ids]
                param_groups_list.append({"params": yolo_params, "lr": self.config.learning_rate * self.config.yolo_lr_scale})
            param_groups: Any = param_groups_list
            _logger.info(
                "Differential LR — BiLSTM: %.2e  ViT fine-tune (%d params): %.2e",
                self.config.learning_rate,
                sum(p.numel() for p in vit_trainable),
                self.config.learning_rate * self.config.vit_lr_scale,
            )
        elif yolo_param_ids and self.config.yolo_lr_scale != 1.0:
            yolo_params = [p for p in model.parameters() if p.requires_grad and id(p) in yolo_param_ids]
            other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in yolo_param_ids]
            param_groups = [
                {"params": other_params, "lr": self.config.learning_rate},
                {"params": yolo_params,  "lr": self.config.learning_rate * self.config.yolo_lr_scale},
            ]
            _logger.info(
                "Differential LR — base: %.2e  YOLO/flow branch (%d params): %.2e",
                self.config.learning_rate,
                sum(p.numel() for p in yolo_params),
                self.config.learning_rate * self.config.yolo_lr_scale,
            )
        else:
            param_groups = model.parameters()

        self.optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # CosineWarmupScheduler when warmup requested; plain cosine otherwise
        total_steps = max(1, self.config.epochs) * 200  # rough estimate
        if self.config.warmup_steps > 0:
            from ttss.training.scheduler import CosineWarmupScheduler
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_steps=self.config.warmup_steps,
                total_steps=total_steps,
                min_lr=self.config.learning_rate * 1e-2,
                last_epoch=-1,
            )
            # Suppress the spurious "step before optimizer" warning on first step.
            # optimizer.step() IS called inside train_step() before scheduler.step().
            # The warning fires because PyTorch's internal counter gets confused by
            # the LRScheduler.__init__ calling step(0) during construction.
            self.scheduler._step_count = 2  # LRScheduler.__init__ calls step() once (→1),
            # making the first manual step() appear as "before optimizer". Advancing to 2
            # tells PyTorch the warning check window has already passed.
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, self.config.epochs))

        # Loss modules
        self._regression_loss  = ThreatScoreRegressionLoss()
        self._consistency_loss = TemporalConsistencyLoss()
        self._precrime_loss    = PreCrimeDetectionLoss(precrime_weight=2.0)
        self._mil_loss         = MILRankingLoss(margin=0.1, top_k=3) if self.config.use_mil_loss else None

        # AMP
        device = next(model.parameters()).device
        self._use_amp = self.config.mixed_precision and device.type == "cuda"
        self._scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler() if self._use_amp else None
        )

        # TensorBoard
        self._global_step: int = 0
        self._tb_writer: Any = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=self.config.tensorboard_dir)
                _logger.info("TensorBoard writer at %s", self.config.tensorboard_dir)
            except ImportError:
                _logger.warning("tensorboard not installed — TB logging disabled")

        _logger.info(
            "TTSSTrainer: epochs=%d lr=%.2e λ_reg=%.2f λ_tc=%.2f λ_pre=%.2f λ_mil=%.2f "
            "warmup=%d amp=%s",
            self.config.epochs, self.config.learning_rate,
            self.config.lambda_reg, self.config.lambda_tc,
            self.config.lambda_pre, self.config.lambda_mil,
            self.config.warmup_steps, self._use_amp,
        )

        # Load frame-level test annotations for proper per-frame GT evaluation
        self._anomaly_spans: dict[str, list[tuple[int, int]]] = {}
        if self.config.val_annotations and os.path.exists(self.config.val_annotations):
            import json
            with open(self.config.val_annotations) as _f:
                _ann_data = json.load(_f)
            for _v in _ann_data["videos"]:
                _spans = [
                    (int(_s["start_frame"]), int(_s["end_frame"]))
                    for _s in _v["anomaly_spans"]
                    if int(_s["start_frame"]) >= 0
                ]
                self._anomaly_spans[_v["video_id"]] = _spans
            _logger.info(
                "Loaded frame-level annotations for %d test videos", len(self._anomaly_spans)
            )

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        if not self.config.use_wandb:
            return
        if os.environ.get("WANDB_MODE") == "disabled":
            return
        try:
            import wandb
            self._wandb_run = wandb.init(project=self.config.wandb_project)
        except ImportError:
            _logger.warning("wandb not installed — logging disabled")

    def _log_wandb(self, metrics: dict[str, float], step: int) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Single training step — ALL losses computed inside the gradient pass
    # ------------------------------------------------------------------

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        precrime_mask: torch.Tensor | None = None,
        accumulate: bool = False,
    ) -> float:
        """Forward + backward + (optionally deferred) optimizer step.

        Args:
            x:             ``(B, T, F)`` feature tensor.
            y:             ``(B, T)`` target threat scores in [0, 1].
            precrime_mask: Optional ``(B, T)`` bool mask for PreCrimeDetectionLoss.
            accumulate:    If True, skip the optimizer step (gradient accumulation).

        Returns:
            Total scalar loss as a Python float.
        """
        self.model.train()
        if not accumulate:
            self.optimizer.zero_grad()

        device = next(self.model.parameters()).device
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device) for t in x)
        else:
            x = x.to(device)
        y = y.to(device)
        if precrime_mask is not None:
            precrime_mask = precrime_mask.to(device)

        amp_ctx: Any = (
            torch.amp.autocast(device_type="cuda")
            if self._use_amp
            else torch.amp.autocast(device_type="cpu", enabled=False)
        )

        accum = self.config.grad_accum_steps

        with amp_ctx:
            result = self.model(x)
            preds = result.frame_scores if hasattr(result, "frame_scores") else result

            reg  = self._regression_loss(preds, y)
            cons = self._consistency_loss(preds)
            loss = self.config.lambda_reg * reg + self.config.lambda_tc * cons

            if precrime_mask is not None and self.config.lambda_pre > 0:
                pre = self._precrime_loss(preds, y, precrime_mask)
                loss = loss + self.config.lambda_pre * pre

            if self._mil_loss is not None and self.config.lambda_mil > 0:
                is_anom = y.max(dim=1).values > 0.5
                if is_anom.any() and (~is_anom).any():
                    mil = self._mil_loss(preds[is_anom], preds[~is_anom])
                    loss = loss + self.config.lambda_mil * mil

            # Scale loss for accumulation so gradients are averaged over accum steps
            scaled_loss = loss / accum

        if self._scaler is not None:
            self._scaler.scale(scaled_loss).backward()
            if not accumulate:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
        else:
            scaled_loss.backward()
            if not accumulate:
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
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        trainer = cls(model, config)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        trainer._best_val_auc = float(ckpt.get("val_auc", 0.0))
        _logger.info(
            "Checkpoint loaded from %s (epoch=%d val_auc=%.4f)",
            path, ckpt.get("epoch", -1), trainer._best_val_auc,
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
        """Run the full training loop with warmup, early stopping, and all metrics."""
        self._init_wandb()
        checkpoint_dir = pathlib.Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs = self.config.dry_run_steps if self.config.dry_run else self.config.epochs
        epoch_losses: list[float] = []
        val_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_steps = 0
            self.model.train()
            total_steps = self.config.steps_per_epoch if self.config.steps_per_epoch > 0 else None
            pbar = tqdm(train_iter, total=total_steps,
                        desc=f"epoch {epoch}/{epochs-1}", unit="step", leave=True)
            step_count = 0
            accum = self.config.grad_accum_steps
            self.optimizer.zero_grad()
            for x, y in pbar:
                step_count += 1
                is_accum_step = (step_count % accum != 0)
                step_loss = self.train_step(x, y, accumulate=is_accum_step)
                if not math.isnan(step_loss):
                    epoch_loss += step_loss
                    n_steps += 1
                pbar.set_postfix(loss=f"{step_loss:.4f}" if not math.isnan(step_loss) else "nan",
                                 avg=f"{epoch_loss/n_steps:.4f}" if n_steps > 0 else "nan")
                if not is_accum_step and self.config.warmup_steps > 0:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        self.scheduler.step()
                if self.config.dry_run and step_count >= self.config.dry_run_steps:
                    break
                if self.config.steps_per_epoch > 0 and step_count >= self.config.steps_per_epoch:
                    break
            pbar.close()

            avg_loss = epoch_loss / max(1, n_steps)
            epoch_losses.append(avg_loss)

            # Epoch-level scheduler step (CosineAnnealing without warmup)
            if self.config.warmup_steps == 0:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # ----------------------------------------------------------
            # Validation: loss + AUC + EAR + MALT + pre-crime AP
            # ----------------------------------------------------------
            val_loss, val_auc, val_ear, val_malt, val_precrime_ap = 0.0, 0.0, 0.0, 0.0, 0.0
            val_acc, val_f1 = 0.0, 0.0
            if val_iter is not None:
                from ttss.training.metrics import (
                    frame_level_auc, early_alert_rate,
                    mean_alert_lead_time, precrime_ap,
                )
                val_device = next(self.model.parameters()).device
                self.model.eval()
                n_val = 0
                all_preds: list[float] = []
                all_targets: list[float] = []
                all_frame_indices_flat: list[int] = []
                all_video_ids_flat: list[str] = []
                with torch.no_grad():
                    for val_batch in val_iter:
                        # Support both 2-tuple (x, y) and 4-tuple (x, y, fidx, vids)
                        if len(val_batch) == 4:
                            x_val, y_val, batch_fidx, batch_vids = val_batch
                        else:
                            x_val, y_val = val_batch
                            batch_fidx, batch_vids = None, None
                        if isinstance(x_val, (tuple, list)):
                            x_val = tuple(t.to(val_device) for t in x_val)
                        else:
                            x_val = x_val.to(val_device)
                        y_val = y_val.to(val_device)
                        result = self.model(x_val)
                        preds = result.frame_scores if hasattr(result, "frame_scores") else result
                        reg  = self._regression_loss(preds, y_val)
                        cons = self._consistency_loss(preds)
                        val_loss += float(
                            (self.config.lambda_reg * reg + self.config.lambda_tc * cons).item()
                        )
                        flat_preds = preds.detach().cpu().reshape(-1).tolist()
                        flat_targets = y_val.detach().cpu().reshape(-1).tolist()
                        all_preds.extend(flat_preds)
                        all_targets.extend(flat_targets)
                        # Collect per-frame metadata for frame-level GT
                        if batch_fidx is not None:
                            T_cur = batch_fidx.shape[1]
                            all_frame_indices_flat.extend(batch_fidx.reshape(-1).tolist())
                            all_video_ids_flat.extend(
                                [vid for vid in batch_vids for _ in range(T_cur)]
                            )
                        n_val += 1
                        if self.config.val_steps > 0 and n_val >= self.config.val_steps:
                            break
                val_loss = val_loss / max(1, n_val)

                if all_preds:
                    y_true_np = np.array(all_targets)
                    y_score_np = np.array(all_preds)

                    # Build frame-level binary GT:
                    # If we have frame_indices + annotation spans, use proper per-frame labels.
                    # Otherwise fall back to clip-level label (all frames in anomaly clip = 1).
                    if all_frame_indices_flat and self._anomaly_spans:
                        y_bin = np.zeros(len(all_frame_indices_flat), dtype=int)
                        for i, (fi, vid) in enumerate(
                            zip(all_frame_indices_flat, all_video_ids_flat)
                        ):
                            if fi < 0:
                                continue  # padding frame — stays 0
                            for s, e in self._anomaly_spans.get(vid, []):
                                if s <= fi <= e:
                                    y_bin[i] = 1
                                    break
                    else:
                        y_bin = (y_true_np >= 0.5).astype(int)
                    if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                        val_auc  = frame_level_auc(y_bin, y_score_np)
                        val_ear  = early_alert_rate(y_bin, y_score_np)
                        val_malt = mean_alert_lead_time(y_bin, y_score_np)
                        # pre-crime AP: frames before first anomaly onset
                        onset_idx = np.where(y_bin == 1)[0]
                        if len(onset_idx) > 0:
                            onset = int(onset_idx[0])
                            y_pre = np.zeros_like(y_bin)
                            y_pre[:onset] = 1
                            if y_pre.sum() > 0:
                                val_precrime_ap = precrime_ap(y_pre, y_score_np)

                    from sklearn.metrics import roc_curve, f1_score, balanced_accuracy_score
                    _fpr, _tpr, _thrs = roc_curve(y_bin, y_score_np)
                    _j = _tpr - _fpr
                    _best_thr = float(_thrs[np.argmax(_j)])
                    y_pred_bin = (y_score_np >= _best_thr).astype(int)
                    val_acc = float(balanced_accuracy_score(y_bin, y_pred_bin))
                    val_f1  = float(f1_score(y_bin, y_pred_bin, zero_division=0))

                self.model.train()

                if val_auc > self._best_val_auc:
                    self._best_val_auc = val_auc
                    self._patience_counter = 0
                    self.save_checkpoint(checkpoint_dir / "best.pt", epoch=epoch, val_auc=val_auc)
                else:
                    self._patience_counter += 1
                    if not self.config.dry_run and self._patience_counter >= self.config.patience:
                        _logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.config.patience)
                        break

            metrics = {
                "train_loss": avg_loss, "val_loss": val_loss,
                "val_auc": val_auc, "val_ear": val_ear,
                "val_malt": val_malt, "val_precrime_ap": val_precrime_ap,
                "val_acc": val_acc, "val_f1": val_f1,
                "lr": current_lr, "epoch": epoch,
            }
            self._log_wandb(metrics, step=epoch)
            _logger.info(
                "epoch=%d  loss=%.4f  val_loss=%.4f  AUC=%.4f  bal_acc=%.4f  F1=%.4f  "
                "EAR=%.4f  MALT=%.1f  pre-AP=%.4f  lr=%.2e",
                epoch, avg_loss, val_loss, val_auc, val_acc, val_f1,
                val_ear, val_malt, val_precrime_ap, current_lr,
            )

            if self._tb_writer is not None:
                self._tb_writer.add_scalar("train/loss_epoch", avg_loss, epoch)
                self._tb_writer.add_scalar("train/lr", current_lr, epoch)
                if val_iter is not None:
                    self._tb_writer.add_scalar("val/loss",        val_loss,        epoch)
                    self._tb_writer.add_scalar("val/auc",         val_auc,         epoch)
                    self._tb_writer.add_scalar("val/bal_acc",      val_acc,         epoch)
                    self._tb_writer.add_scalar("val/f1",          val_f1,          epoch)
                    self._tb_writer.add_scalar("val/ear",         val_ear,         epoch)
                    self._tb_writer.add_scalar("val/malt",        val_malt,        epoch)
                    self._tb_writer.add_scalar("val/precrime_ap", val_precrime_ap, epoch)

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
            val_loss=val_loss,
            best_val_auc=self._best_val_auc,
            epochs_completed=len(epoch_losses),
            metrics={
                "train_loss": train_loss, "val_loss": val_loss,
                "val_auc": val_auc, "val_ear": val_ear,
                "val_malt": val_malt, "val_precrime_ap": val_precrime_ap,
                "val_acc": val_acc, "val_f1": val_f1,
            },
        )
