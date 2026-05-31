"""Temporal Threat Scoring System (TTSS): training CLI.

Two modes:
  --dry-run      Trains on synthetic random tensors (verifies the pipeline, no data needed)
  (default)      Trains on pre-extracted features from data/features/

Full workflow::

    # 1. Download UCF-Crime (runs automatically if Dropbox is accessible)
    # 2. Unzip
    python -m ttss.scripts.unzip_dataset

    # 3. Extract YOLOv8m + ViT features for all videos (~10h on RTX 4050)
    python -m ttss.scripts.extract_features --device cuda

    # 4. Train
    python -m ttss.scripts.train --config ttss/configs/base.yaml

    # Quick smoke test (no data required)
    python -m ttss.scripts.train --dry-run
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import random

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")

from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor
from ttss.models.fusion.object_scene_attention import ObjectConditionedThreatModel
from ttss.training.reproducibility import RunConfig, save_run_config, seed_everything
from ttss.training.trainer import TTSSTrainer, TrainerConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the TTSS BiLSTM threat predictor on pre-extracted features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="ttss/configs/base.yaml")
    parser.add_argument("--features-dir", default="data/features",
                        help="Root of pre-extracted feature files (train/ and test/ subdirs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train on synthetic data without real features")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vit-unfreeze-blocks", type=int, default=None)
    parser.add_argument(
        "--resume", metavar="CHECKPOINT", default=None,
        help="Path to a .pt checkpoint. Loads model weights only; optimizer and "
             "scheduler are reset so the new config's LR/epochs apply from epoch 0.",
    )
    return parser


# ---------------------------------------------------------------------------
# Data iterators
# ---------------------------------------------------------------------------


def _real_iter(
    split: str,
    features_dir: str,
    clip_length: int,
    batch_size: int,
    shuffle: bool = True,
):
    """Yield (features, labels) batches from pre-extracted .npz files.

    Features: (B, T, 776)  float32
    Labels:   (B, T)       float32  — 1.0 for anomaly clips, 0.0 for normal
    """
    from ttss.data.feature_dataset import FeatureDataset, mil_collate_fn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = FeatureDataset(features_dir, clip_length=clip_length, split=split)
    if len(ds) == 0:
        raise RuntimeError(
            f"No .npz feature files found under {features_dir}/{split}/\n"
            "Run: python -m ttss.scripts.extract_features --device cuda"
        )

    # Balanced sampling: equal anomaly / normal batches
    anomaly_idx = ds.anomaly_indices()
    normal_idx  = ds.normal_indices()
    n_a, n_n = len(anomaly_idx), len(normal_idx)
    print(f"  {split}: {len(ds)} clips  ({n_a} anomaly, {n_n} normal)")

    if n_a > 0 and n_n > 0:
        weights = [2.0 / n_a if ds._files[i].stem.replace("_x264","") not in
                   {"Normal_Videos","Normal","normal"} else 1.0 / n_n
                   for i in range(len(ds))]
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                            collate_fn=mil_collate_fn, num_workers=4, pin_memory=True,
                            persistent_workers=True)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=mil_collate_fn, num_workers=4, pin_memory=True,
                            persistent_workers=True)

    while True:
        for batch in loader:
            yield (batch["yolo"], batch["vit"], batch["flow"]), batch["labels"]


def _real_val_iter(
    features_dir: str,
    clip_length: int,
    batch_size: int,
):
    """Yield (x, labels, frame_indices, video_ids) 4-tuples for the test split.

    Includes frame_indices and video_ids so the trainer can compute proper
    frame-level evaluation using temporal annotations.
    """
    from ttss.data.feature_dataset import FeatureDataset, mil_collate_fn
    from torch.utils.data import DataLoader

    ds = FeatureDataset(features_dir, clip_length=clip_length, split="test")
    if len(ds) == 0:
        raise RuntimeError(
            f"No .npz feature files found under {features_dir}/test/\n"
            "Run: python -m ttss.scripts.extract_features --device cuda"
        )
    print(f"  test: {len(ds)} clips", flush=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=mil_collate_fn, num_workers=4, pin_memory=True,
                        persistent_workers=True)
    while True:
        for batch in loader:
            yield (
                (batch["yolo"], batch["vit"], batch["flow"]),
                batch["labels"],
                batch["frame_indices"],
                batch["video_ids"],
            )


def _synthetic_iter(B: int, T: int, F: int = 776, n_batches: int = 0):
    """Synthetic iterator. Infinite when n_batches=0, finite otherwise.

    Yields 3-tuples (yolo, vmae, flow) matching the enriched feature dims.
    F is ignored; kept for backward-compat signature only.
    """
    count = 0
    while True:
        yolo = torch.rand(B, T, 32)
        vmae = torch.rand(B, T, 768)
        flow = torch.rand(B, T, 16)
        y = torch.cat([torch.zeros(B, T // 2), torch.rand(B, T - T // 2) * 0.8 + 0.2], dim=1)
        yield (yolo, vmae, flow), y
        count += 1
        if n_batches > 0 and count >= n_batches:
            return




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    cfg = {}
    config_path = pathlib.Path(args.config)
    if config_path.exists():
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}

    seed_everything(args.seed)

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg  = cfg.get("data", {})
    log_cfg   = cfg.get("logging", {})

    device      = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    clip_length = data_cfg.get("clip_length", 64)
    num_unfreeze = (args.vit_unfreeze_blocks if args.vit_unfreeze_blocks is not None
                    else model_cfg.get("vit_unfreeze_blocks", 0))

    config = TrainerConfig(
        epochs          = args.epochs        or train_cfg.get("epochs",         30),
        learning_rate   = args.learning_rate  or train_cfg.get("learning_rate",  1e-4),
        batch_size      = args.batch_size    or train_cfg.get("batch_size",     8),
        weight_decay    = train_cfg.get("weight_decay",    1e-5),
        lambda_reg      = train_cfg.get("lambda_reg",      1.0),
        lambda_tc       = train_cfg.get("lambda_tc",       0.1),
        lambda_pre      = train_cfg.get("lambda_pre",      0.5),
        lambda_mil      = train_cfg.get("lambda_mil",      1.0),
        lambda1         = train_cfg.get("lambda_reg",      1.0),
        lambda2         = train_cfg.get("lambda_tc",       0.1),
        max_grad_norm   = train_cfg.get("grad_clip",       1.0),
        patience        = train_cfg.get("patience",        5),
        warmup_steps    = train_cfg.get("warmup_steps",    100),
        grad_accum_steps= train_cfg.get("grad_accum_steps", 1),
        mixed_precision = train_cfg.get("mixed_precision", True),
        vit_lr_scale    = train_cfg.get("vit_lr_scale",    0.1),
        yolo_lr_scale   = train_cfg.get("yolo_lr_scale",   1.0),
        use_wandb       = log_cfg.get("use_wandb",  False),
        wandb_project   = log_cfg.get("project",    "ttss"),
        checkpoint_dir  = str(pathlib.Path(cfg.get("experiment", {}).get("output_dir", "outputs/ttss")) / "checkpoints"),
        dry_run         = args.dry_run,
    )

    print(f"Device: {device}  |  Clip: {clip_length}  |  Batch: {config.batch_size}  |  Seed: {args.seed}")

    # Save run config
    out_dir = pathlib.Path(cfg.get("experiment", {}).get("output_dir", "outputs/ttss"))
    run_cfg = RunConfig.from_yaml_config(cfg, experiment_name=cfg.get("experiment", {}).get("name", "ttss-run"), seed=args.seed)
    save_run_config(run_cfg, out_dir / "run_config.yaml")
    print(f"Run config → {out_dir / 'run_config.yaml'}  (git={run_cfg.git_commit})")

    # Object-Conditioned Cross-Attention + BiLSTM
    # YOLO (8-dim) and VideoMAE (768-dim) are fused via bidirectional cross-attention
    # before being fed to the BiLSTM threat predictor.
    vmae_dim = model_cfg.get("vmae_dim", 1024)  # 1024 for Large, 768 for Base
    model = ObjectConditionedThreatModel(
        yolo_dim=32, vmae_dim=vmae_dim, flow_dim=16, hidden_dim=256,
        num_heads=4, dropout=0.1,
        bilstm_layers=2, bilstm_hidden=256,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"BiLSTM params: {total_params / 1e6:.2f}M")

    trainer = TTSSTrainer(model, config)

    if args.resume:
        ckpt_path = pathlib.Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        prev_auc = float(ckpt.get("val_auc", 0.0))
        prev_epoch = int(ckpt.get("epoch", -1))
        trainer._best_val_auc = prev_auc  # preserve baseline so only genuinely better checkpoints overwrite best.pt
        print(f"Resumed model weights from {ckpt_path} "
              f"(epoch={prev_epoch} val_auc={prev_auc:.4f}) — optimizer/scheduler reset")

    if not args.dry_run:
        print(f"MIL ranking loss: enabled (λ={config.lambda_mil})")

    # Data iterators
    features_dir = args.features_dir
    B = config.batch_size

    if args.dry_run:
        print("DRY RUN — using synthetic data")
        train_it = _synthetic_iter(B, clip_length)
        val_it   = _synthetic_iter(B, clip_length, n_batches=3)
    else:
        train_dir = pathlib.Path(features_dir) / "train"
        val_dir   = pathlib.Path(features_dir) / "test"

        if not train_dir.exists():
            print(f"\nFeature directory not found: {train_dir}")
            print("Run feature extraction first:")
            print("  python -m ttss.scripts.extract_features --device cuda")
            return

        from ttss.data.feature_dataset import FeatureDataset
        _train_ds = FeatureDataset(features_dir, clip_length=clip_length, split="train")
        steps_per_epoch = max(1, len(_train_ds) // B)
        config.steps_per_epoch = steps_per_epoch
        print(f"  steps_per_epoch={steps_per_epoch} ({len(_train_ds)} train clips, batch={B})")

        _val_ds = FeatureDataset(features_dir, clip_length=clip_length, split="test")
        config.val_steps = max(1, len(_val_ds) // B)

        train_it = _real_iter("train", features_dir, clip_length, B, shuffle=True)
        val_it   = _real_val_iter(features_dir, clip_length, B)

        # Frame-level annotations for proper evaluation
        annotations_path = str(
            pathlib.Path(features_dir).parent / "raw/UCF-Crime/annotations/test_annotations.json"
        )
        if pathlib.Path(annotations_path).exists():
            config.val_annotations = annotations_path
            print(f"  Frame-level annotations: {annotations_path}")

    result = trainer.fit(train_it, val_iter=val_it)
    print(f"\ntrain_loss={result.train_loss:.4f}  val_loss={result.val_loss:.4f}  "
          f"best_val_auc={result.best_val_auc:.4f}  epochs={result.epochs_completed}")


if __name__ == "__main__":
    main()
