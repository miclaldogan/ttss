"""Temporal Threat Scoring System (TTSS): package training CLI."""

from __future__ import annotations

import argparse

from ttss.models.ttss_pipeline import TtssPipeline
from ttss.training.trainer import Trainer, TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the TTSS training CLI parser."""
    parser = argparse.ArgumentParser(description="Train the TTSS research stack.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    return parser


def main() -> None:
    """Run the TTSS training scaffold."""
    args = build_parser().parse_args()
    trainer = Trainer(
        pipeline=TtssPipeline(),
        config=TrainerConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        ),
    )
    result = trainer.fit(train_batches=[[]], train_targets=[0.0])
    print(f"train_loss={result.train_loss:.4f}")


if __name__ == "__main__":
    main()
