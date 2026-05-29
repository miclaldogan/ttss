"""Temporal Threat Scoring System (TTSS): learning rate schedulers."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupScheduler(LRScheduler):
    """Cosine annealing with a linear warmup phase.

    For the first *warmup_steps* steps the LR rises linearly from 0 to
    ``base_lr``.  After warmup it follows a cosine decay to ``min_lr``.

    Parameters
    ----------
    optimizer:     Wrapped optimizer.
    warmup_steps:  Number of steps (not epochs) for the linear warmup.
    total_steps:   Total training steps (warmup + cosine decay).
    min_lr:        Floor LR at the end of cosine decay (default 0).
    last_epoch:    Last completed step index (default -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                scale = (step + 1) / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))
                scale = self.min_lr / base_lr + (1.0 - self.min_lr / base_lr) * scale
            lrs.append(base_lr * scale)
        return lrs
