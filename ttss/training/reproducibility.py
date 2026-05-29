"""Temporal Threat Scoring System (TTSS): reproducibility utilities.

Three components:

1. seed_everything(seed)    — deterministic seeding across torch, numpy,
                               random, and CUDA.
2. RunConfig                — captures all hyperparams + git commit hash +
                               timestamp; serialisable to YAML.
3. save_run_config(cfg, path) — write RunConfig to disk as YAML so that every
                               experiment is fully reproducible from its log.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# 1. Seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    """Seed all relevant RNGs for fully deterministic training.

    Sets:
    * ``random`` (Python stdlib)
    * ``numpy``
    * ``torch`` (CPU and CUDA)
    * ``PYTHONHASHSEED`` environment variable
    * ``torch.backends.cudnn.deterministic = True``
    * ``torch.backends.cudnn.benchmark = False``

    .. warning::
        ``cudnn.deterministic = True`` can slow down training by 10-20% on
        some architectures.  Set ``cudnn_deterministic=False`` if speed is
        critical and exact bit-for-bit reproducibility across runs is not
        required.

    Parameters
    ----------
    seed:
        Integer seed value.  Use the same seed across all workers and
        dataloaders (set ``worker_init_fn`` in DataLoader accordingly).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """DataLoader ``worker_init_fn`` that seeds each worker deterministically.

    Pass as ``DataLoader(worker_init_fn=lambda wid: worker_init_fn(wid, seed))``.
    """
    seed_everything(base_seed + worker_id)


# ---------------------------------------------------------------------------
# 2. RunConfig
# ---------------------------------------------------------------------------


def _git_commit_hash() -> str:
    """Return the current HEAD commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_diff_hash() -> str:
    """Return an MD5 hash of any uncommitted changes, or 'clean'."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, timeout=5,
        )
        diff = result.stdout.strip()
        if not diff:
            return "clean"
        return hashlib.md5(diff.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


@dataclass
class RunConfig:
    """Complete run specification for reproducibility logging.

    Captures all hyperparameters, model settings, dataset paths, and
    environment metadata in a single serialisable object.  Write to disk
    with :func:`save_run_config` at the start of each training run.

    Fields
    ------
    experiment_name:
        Human-readable run label (used as TensorBoard / W&B run name).
    seed:
        Global random seed passed to :func:`seed_everything`.
    git_commit:
        Short HEAD commit hash at run start.
    git_diff_hash:
        MD5 of ``git diff`` output — ``'clean'`` when the working tree is clean.
    timestamp:
        ISO-8601 UTC timestamp of run creation.
    training:
        Flat dict of training hyperparameters (epochs, lr, batch_size, etc.).
    model:
        Model architecture settings (yolo_variant, vit_unfreeze_blocks, etc.).
    data:
        Dataset settings (data_root, frame_stride, clip_length, split_file, etc.).
    extra:
        Arbitrary additional metadata.
    """

    experiment_name: str = "ttss-run"
    seed: int = 42
    git_commit: str = field(default_factory=_git_commit_hash)
    git_diff_hash: str = field(default_factory=_git_diff_hash)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    training: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml_config(cls, config: dict, experiment_name: str = "ttss-run", seed: int = 42) -> "RunConfig":
        """Build a RunConfig from the standard TTSS YAML config dict."""
        return cls(
            experiment_name=experiment_name,
            seed=seed,
            training=config.get("training", {}),
            model=config.get("model", {}),
            data={k: v for k, v in config.get("data", {}).items()
                  if k not in ("download_url", "archive_path")},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 3. Save / load
# ---------------------------------------------------------------------------


def save_run_config(cfg: RunConfig, path: str | Path) -> Path:
    """Serialise *cfg* to a YAML file at *path*.

    Creates parent directories automatically.  Returns the resolved path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)
    return out


def load_run_config(path: str | Path) -> RunConfig:
    """Deserialise a YAML run config back into a :class:`RunConfig`."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunConfig(**{k: v for k, v in data.items() if k in RunConfig.__dataclass_fields__})


# ---------------------------------------------------------------------------
# UCF-Crime standard split
# ---------------------------------------------------------------------------

# Official UCF-Crime 80/20 split — anomaly categories only.
# Normal videos are split 800 train / 150 test following the standard protocol.
UCF_CRIME_SPLIT: dict[str, list[str]] = {
    "anomaly_train_ratio": [0.8],
    "normal_train": [800],
    "normal_test": [150],
    "categories": [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism",
    ],
}


def make_ucf_crime_split(
    annotation_records,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> tuple[list, list]:
    """Split UCF-Crime annotation records into train/test sets.

    Stratifies by crime category so each category has the same train ratio.
    Normal videos receive the same overall split ratio.

    Parameters
    ----------
    annotation_records:
        List of ``AnnotationRecord`` (or any object with a ``.label`` attribute).
    seed:
        Random seed for reproducible splits.
    train_ratio:
        Fraction of each category to use for training.

    Returns
    -------
    (train_records, test_records)
    """
    rng = random.Random(seed)

    # Group by label
    groups: dict[str, list] = {}
    for rec in annotation_records:
        groups.setdefault(rec.label, []).append(rec)

    train, test = [], []
    for label, recs in groups.items():
        shuffled = list(recs)
        rng.shuffle(shuffled)
        n_train = max(1, round(len(shuffled) * train_ratio))
        train.extend(shuffled[:n_train])
        test.extend(shuffled[n_train:])

    return train, test
