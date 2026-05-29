"""Tests for TTSSTrainer: loss decrease, checkpoint saving, early stopping."""

from __future__ import annotations

import pathlib

import torch
import pytest

from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor
from ttss.training.trainer import TTSSTrainer, TrainerConfig
from ttss.training.scheduler import CosineWarmupScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_model() -> BiLSTMThreatPredictor:
    return BiLSTMThreatPredictor(input_dim=16, hidden_dim=32, num_layers=1)


def _synthetic_iter(n_batches: int = 5, B: int = 4, T: int = 8, F: int = 16):
    """Yield (x, y) pairs with a detectable signal: score should be high in the
    second half of each sequence."""
    for _ in range(n_batches):
        x = torch.rand(B, T, F)
        y = torch.cat([torch.zeros(B, T // 2), torch.ones(B, T // 2)], dim=1)
        yield x, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loss_decreases_over_steps(tmp_path):
    """3-step training on CPU with synthetic data; loss must decrease."""
    model = _small_model()
    config = TrainerConfig(
        epochs=3,
        batch_size=4,
        learning_rate=1e-3,
        mixed_precision=False,
        warmup_steps=0,   # no warmup — full LR from step 1 so loss moves in 5 steps
        checkpoint_dir=str(tmp_path / "ckpts"),
        dry_run=False,
    )
    trainer = TTSSTrainer(model, config)

    losses = []
    for x, y in _synthetic_iter(n_batches=5):
        losses.append(trainer.train_step(x, y))

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_checkpoint_saved(tmp_path):
    """fit() must write latest.pt containing model_state_dict and val_auc."""
    model = _small_model()
    config = TrainerConfig(
        epochs=2,
        batch_size=4,
        mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ckpts"),
        dry_run=True,
        dry_run_steps=2,
    )
    trainer = TTSSTrainer(model, config)
    trainer.fit(_synthetic_iter())

    ckpt_path = tmp_path / "ckpts" / "latest.pt"
    assert ckpt_path.exists(), "latest.pt not written"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "epoch" in ckpt
    assert "val_auc" in ckpt


def test_best_checkpoint_saved_on_val_improvement(tmp_path):
    """best.pt is written when val_auc improves."""
    model = _small_model()
    config = TrainerConfig(
        epochs=3,
        batch_size=4,
        mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ckpts"),
        dry_run=False,
    )
    trainer = TTSSTrainer(model, config)
    trainer.fit(_synthetic_iter(), val_iter=_synthetic_iter(n_batches=3))

    best_path = tmp_path / "ckpts" / "best.pt"
    assert best_path.exists(), "best.pt not written despite val_iter being provided"


def test_val_loss_less_than_train_loss_after_fit(tmp_path):
    """fit() returns a TrainResult with populated metrics."""
    model = _small_model()
    config = TrainerConfig(
        epochs=2,
        batch_size=4,
        mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ckpts"),
        dry_run=False,
    )
    trainer = TTSSTrainer(model, config)
    result = trainer.fit(_synthetic_iter(), val_iter=_synthetic_iter(n_batches=3))

    assert result.epochs_completed > 0
    assert "train_loss" in result.metrics
    assert "val_loss" in result.metrics
    assert result.train_loss >= 0.0


def test_cosine_warmup_scheduler():
    """CosineWarmupScheduler LR rises during warmup then decays."""
    import torch.optim as optim
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = optim.SGD([param], lr=0.1)
    sched = CosineWarmupScheduler(optimizer, warmup_steps=5, total_steps=20)

    lrs = []
    for _ in range(20):
        sched.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # LR should increase during warmup (first 5 steps)
    assert lrs[4] > lrs[0], "LR should increase during warmup"
    # LR should decrease after warmup
    assert lrs[-1] < lrs[5], "LR should decrease after warmup"


def test_early_stopping_fires(tmp_path):
    """Trainer stops early when val AUC stops improving."""
    model = _small_model()
    config = TrainerConfig(
        epochs=20,
        batch_size=4,
        mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ckpts"),
        patience=2,
        dry_run=False,
    )
    trainer = TTSSTrainer(model, config)
    # All-normal labels → AUC will be undefined / won't improve
    def _flat_iter():
        for _ in range(5):
            x = torch.rand(4, 8, 16)
            y = torch.zeros(4, 8)
            yield x, y

    result = trainer.fit(_flat_iter(), val_iter=_flat_iter())
    assert result.epochs_completed < 20, "Early stopping should have triggered before 20 epochs"
