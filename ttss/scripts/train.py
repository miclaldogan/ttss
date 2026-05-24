"""Temporal Threat Scoring System (TTSS): package training CLI."""

from __future__ import annotations

import argparse
import pathlib

import torch
import yaml

from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor
from ttss.training.trainer import TTSSTrainer, TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the TTSS training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train the TTSS research stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="ttss/configs/base.yaml", metavar="PATH",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 2 synthetic steps to verify the pipeline",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override LR")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser


def _load_config(config_path: str) -> dict:
    path = pathlib.Path(config_path)
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    """Run the TTSS training scaffold."""
    args = build_parser().parse_args()
    cfg = _load_config(args.config)
    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})

    config = TrainerConfig(
        epochs=args.epochs or train_cfg.get("epochs", 20),
        learning_rate=args.learning_rate or train_cfg.get("learning_rate", 1e-4),
        batch_size=args.batch_size or train_cfg.get("batch_size", 8),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("project", "ttss"),
        dry_run=args.dry_run,
    )

    model = BiLSTMThreatPredictor(input_dim=1536)
    trainer = TTSSTrainer(model, config)

    # Synthetic data — used for dry-run and as a fallback
    T, F = 16, 1536
    B = config.batch_size

    def _synthetic_iter():
        while True:
            yield (
                torch.rand(B, T, F),
                torch.rand(B, T),
            )

    result = trainer.fit(_synthetic_iter())
    print(f"train_loss={result.train_loss:.4f}")
    print(f"epochs_completed={result.epochs_completed}")


if __name__ == "__main__":
    main()

