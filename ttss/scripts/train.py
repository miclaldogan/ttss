"""Temporal Threat Scoring System (TTSS): package training CLI."""

from __future__ import annotations

import argparse
import pathlib

import torch
import yaml

from ttss.models.end_to_end import EndToEndThreatModel
from ttss.training.reproducibility import RunConfig, save_run_config, seed_everything
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
    parser.add_argument(
        "--vit-unfreeze-blocks", type=int, default=None,
        help="Override number of ViT tail blocks to fine-tune (0 = fully frozen)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
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
    model_cfg = cfg.get("model", {})
    log_cfg = cfg.get("logging", {})

    seed_everything(args.seed)
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    num_unfreeze = args.vit_unfreeze_blocks if args.vit_unfreeze_blocks is not None \
        else model_cfg.get("vit_unfreeze_blocks", 2)
    use_yolo = model_cfg.get("use_yolo_features", True)

    config = TrainerConfig(
        epochs=args.epochs or train_cfg.get("epochs", 30),
        learning_rate=args.learning_rate or train_cfg.get("learning_rate", 1e-4),
        batch_size=args.batch_size or train_cfg.get("batch_size", 4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        vit_lr_scale=train_cfg.get("vit_lr_scale", 0.1),
        mixed_precision=train_cfg.get("mixed_precision", True),
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("project", "ttss"),
        dry_run=args.dry_run,
    )

    print(f"Device: {device} | ViT unfreeze blocks: {num_unfreeze} | Mixed precision: {config.mixed_precision} | Seed: {args.seed}")

    run_cfg = RunConfig.from_yaml_config(cfg, experiment_name=cfg.get("experiment", {}).get("name", "ttss-run"), seed=args.seed)
    out_dir = pathlib.Path(cfg.get("experiment", {}).get("output_dir", "outputs/ttss"))
    save_run_config(run_cfg, out_dir / "run_config.yaml")
    print(f"Run config saved → {out_dir / 'run_config.yaml'}  (git={run_cfg.git_commit})")

    model = EndToEndThreatModel.build(
        num_unfreeze_blocks=num_unfreeze,
        device=device,
        use_yolo_features=use_yolo,
    )
    trainer = TTSSTrainer(model, config)

    # Synthetic data — shape matches the end-to-end model:
    # x: (B, T, 3, 224, 224) preprocessed frames
    # yolo_feats: (B, T, 8) pre-computed YOLO features  (packed into x as tuple)
    # y: (B, T) threat score labels in [0, 1]
    B = config.batch_size
    T = cfg.get("data", {}).get("clip_length", 64)

    def _synthetic_iter():
        while True:
            frames = torch.rand(B, T, 3, 224, 224, device=device)
            labels = torch.rand(B, T, device=device)
            if use_yolo:
                yolo_feats = torch.rand(B, T, 8, device=device)
                yield (frames, yolo_feats), labels
            else:
                yield frames, labels

    result = trainer.fit(_synthetic_iter())
    print(f"train_loss={result.train_loss:.4f}")
    print(f"epochs_completed={result.epochs_completed}")


if __name__ == "__main__":
    main()
