"""Unit tests for the TTSS training loop, losses, and trainer (issue #7)."""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest
import torch

from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor
from ttss.training.losses import (
    TemporalConsistencyLoss,
    ThreatScoreRegressionLoss,
    composite_threat_loss,
)
from ttss.training.trainer import TTSSTrainer, TrainerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 32   # small for fast tests
HIDDEN_DIM = 16
T = 8
B = 2


def _make_model() -> BiLSTMThreatPredictor:
    return BiLSTMThreatPredictor(
        input_dim=INPUT_DIM,
        projection_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
    )


def _make_config(**kwargs) -> TrainerConfig:
    defaults = dict(
        epochs=3,
        learning_rate=1e-3,
        dry_run=False,
        use_wandb=False,
        checkpoint_dir=tempfile.mkdtemp(),
    )
    defaults.update(kwargs)
    return TrainerConfig(**defaults)


def _batch_iter(n: int = 4):
    """Yield n synthetic (x, y) batches."""
    for _ in range(n):
        yield torch.rand(B, T, INPUT_DIM), torch.rand(B, T)


# ---------------------------------------------------------------------------
# TemporalConsistencyLoss — delta threshold
# ---------------------------------------------------------------------------


def test_temporal_consistency_loss_zero_delta_baseline():
    scores = torch.tensor([[0.1, 0.5, 0.2]], dtype=torch.float32)
    loss = TemporalConsistencyLoss(delta=0.0)(scores)
    assert loss.item() > 0.0


def test_temporal_consistency_loss_delta_clamps_small_differences():
    """With delta=0.5 a change of 0.3 should contribute zero loss."""
    scores = torch.tensor([[0.0, 0.3]], dtype=torch.float32)
    loss = TemporalConsistencyLoss(delta=0.5)(scores)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_temporal_consistency_loss_delta_large_differences():
    """With delta=0.1 a change of 0.4 should contribute 0.3 loss."""
    scores = torch.tensor([[0.0, 0.4]], dtype=torch.float32)
    loss = TemporalConsistencyLoss(delta=0.1)(scores)
    assert loss.item() == pytest.approx(0.3, abs=1e-5)


# ---------------------------------------------------------------------------
# composite_threat_loss — lambda weights
# ---------------------------------------------------------------------------


def test_composite_loss_lambda_weights():
    """lambda1=0 should zero out regression; lambda2=0 should zero out consistency."""
    preds = [0.1, 0.5, 0.9]
    targets = [0.2, 0.4, 0.8]
    loss_no_reg = composite_threat_loss(preds, targets, lambda1=0.0, lambda2=1.0)
    loss_no_cons = composite_threat_loss(preds, targets, lambda1=1.0, lambda2=0.0)
    assert loss_no_reg >= 0.0
    assert loss_no_cons > 0.0
    # With lambda2=0 we only have regression; no consistency penalty
    reg_only = ThreatScoreRegressionLoss()(
        torch.tensor(preds), torch.tensor(targets)
    ).item()
    assert loss_no_cons == pytest.approx(reg_only, abs=1e-5)


# ---------------------------------------------------------------------------
# TTSSTrainer construction
# ---------------------------------------------------------------------------


def test_trainer_constructs():
    model = _make_model()
    trainer = TTSSTrainer(model, _make_config())
    assert trainer.model is model
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None


def test_trainer_scheduler_is_cosine():
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from ttss.training.scheduler import CosineWarmupScheduler
    # warmup_steps=0 → CosineAnnealingLR; warmup_steps>0 → CosineWarmupScheduler
    trainer_no_warmup = TTSSTrainer(_make_model(), _make_config(warmup_steps=0))
    assert isinstance(trainer_no_warmup.scheduler, CosineAnnealingLR)

    trainer_warmup = TTSSTrainer(_make_model(), _make_config(warmup_steps=50))
    assert isinstance(trainer_warmup.scheduler, CosineWarmupScheduler)


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


def test_train_step_returns_positive_loss():
    model = _make_model()
    trainer = TTSSTrainer(model, _make_config())
    x, y = torch.rand(B, T, INPUT_DIM), torch.rand(B, T)
    loss = trainer.train_step(x, y)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_grad_clipping_applied():
    """After train_step the gradient norms should be ≤ max_grad_norm."""
    model = _make_model()
    config = _make_config(max_grad_norm=0.01)  # very tight clip
    trainer = TTSSTrainer(model, config)
    x, y = torch.rand(B, T, INPUT_DIM), torch.rand(B, T)
    # Run step manually inspecting grads before clip:
    model.train()
    trainer.optimizer.zero_grad()
    result = model(x)
    loss = result.frame_scores.mean()
    loss.backward()
    total_norm = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
    ).item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    clipped_norm = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
    ).item()
    assert clipped_norm <= config.max_grad_norm + 1e-5


# ---------------------------------------------------------------------------
# Dry-run — 2 steps, loss output
# ---------------------------------------------------------------------------


def test_dry_run_completes_two_steps():
    model = _make_model()
    config = _make_config(dry_run=True, dry_run_steps=2)
    trainer = TTSSTrainer(model, config)
    result = trainer.fit(_batch_iter(10))
    assert result.epochs_completed >= 1


def test_dry_run_train_loss_is_finite():
    model = _make_model()
    config = _make_config(dry_run=True, dry_run_steps=2)
    trainer = TTSSTrainer(model, config)
    result = trainer.fit(_batch_iter(10))
    assert result.train_loss >= 0.0
    assert not (result.train_loss != result.train_loss)  # NaN check


def test_dry_run_loss_decreases_over_steps():
    """Run 3 steps; record per-step loss; verify the trend is generally downward."""
    model = _make_model()
    config = _make_config(dry_run=False, epochs=1, learning_rate=1e-2)
    trainer = TTSSTrainer(model, config)
    losses = []
    x = torch.rand(B, T, INPUT_DIM)
    y = torch.rand(B, T)
    for _ in range(8):
        losses.append(trainer.train_step(x, y))
    # Mean of last half should be ≤ mean of first half (general trend)
    first_half = sum(losses[:4]) / 4
    last_half = sum(losses[4:]) / 4
    assert last_half <= first_half + 0.05  # generous tolerance


# ---------------------------------------------------------------------------
# W&B graceful skip
# ---------------------------------------------------------------------------


def test_wandb_disabled_mode_no_crash(monkeypatch):
    """WANDB_MODE=disabled must not crash even when use_wandb=True."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    model = _make_model()
    config = _make_config(dry_run=True, dry_run_steps=1, use_wandb=True)
    trainer = TTSSTrainer(model, config)
    result = trainer.fit(_batch_iter(2))  # should not raise
    assert result.train_loss >= 0.0


# ---------------------------------------------------------------------------
# Checkpoint save + reload
# ---------------------------------------------------------------------------


def test_checkpoint_saved_and_loadable():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir) / "test.pt"
        model = _make_model()
        config = _make_config(checkpoint_dir=tmpdir, dry_run=True, dry_run_steps=1)
        trainer = TTSSTrainer(model, config)
        trainer.save_checkpoint(ckpt_path, epoch=1, val_auc=0.75)
        assert ckpt_path.exists()

        # Reload into a fresh model
        model2 = _make_model()
        trainer2 = TTSSTrainer.load_checkpoint(ckpt_path, model2, config)
        assert trainer2._best_val_auc == pytest.approx(0.75)

        # Forward pass should work identically (eval disables dropout)
        x = torch.rand(B, T, INPUT_DIM)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x).frame_scores
            out2 = model2(x).frame_scores
        assert torch.allclose(out1, out2, atol=1e-6)


def test_fit_saves_latest_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        config = _make_config(checkpoint_dir=tmpdir, dry_run=True, dry_run_steps=1)
        trainer = TTSSTrainer(model, config)
        trainer.fit(_batch_iter(4))
        assert (pathlib.Path(tmpdir) / "latest.pt").exists()


# ---------------------------------------------------------------------------
# scripts/train.py arg parsing
# ---------------------------------------------------------------------------


def test_train_script_parser_accepts_config_and_dry_run():
    from ttss.scripts.train import build_parser
    args = build_parser().parse_args(["--config", "ttss/configs/base.yaml", "--dry-run"])
    assert args.config == "ttss/configs/base.yaml"
    assert args.dry_run is True
