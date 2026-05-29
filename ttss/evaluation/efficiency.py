"""Temporal Threat Scoring System (TTSS): inference efficiency utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LatencyResult:
    """Profiling output for a single model component."""

    name: str
    mean_ms: float
    std_ms: float
    p95_ms: float
    fps: float
    n_runs: int


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters broken down by sub-module.

    Returns a dict with keys ``total``, ``trainable``, and one entry per
    named top-level child (e.g. ``recognition``, ``detection``, ``prediction``).
    """
    result: dict[str, int] = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    for name, child in model.named_children():
        result[name] = sum(p.numel() for p in child.parameters())
    return result


def profile_forward(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 100,
    name: str = "model",
) -> LatencyResult:
    """Profile a single forward pass over *n_runs* iterations.

    Parameters
    ----------
    model:        Model to profile (set to eval mode automatically).
    input_shape:  Shape of a single input tensor (batch dimension included).
    device:       ``'cpu'`` or ``'cuda'``.
    n_warmup:     Ignored warm-up iterations.
    n_runs:       Timed iterations.
    name:         Label for the result.

    Returns
    -------
    :class:`LatencyResult` with mean, std, 95th-pct latency in ms and FPS.
    """
    model.eval()
    x = torch.rand(*input_shape, device=device)

    def _sync():
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

    timings: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            _sync()
            t0 = time.perf_counter()
            model(x)
            _sync()
            timings.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(timings)
    batch = input_shape[0] if len(input_shape) > 0 else 1
    return LatencyResult(
        name=name,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        p95_ms=float(np.percentile(arr, 95)),
        fps=batch * 1000.0 / float(arr.mean()),
        n_runs=n_runs,
    )
