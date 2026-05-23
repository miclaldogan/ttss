"""Unit tests for the ViT-B/16 scene encoder (detection layer).

All tests use an injected fake backbone so no network access or GPU is required.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from ttss.models.detection.vit_scene import SceneEmbedding, VitSceneEncoder

EMBEDDING_DIM = 768
BATCH_SIZE = 4
IMG_SIZE = 224


class _FakeViT(nn.Module):
    """Minimal stand-in for a timm ViT model (no weights download needed)."""

    def __init__(self, embed_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._linear = nn.Linear(3 * IMG_SIZE * IMG_SIZE, embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Returns shape (B, 1+num_patches, D) — index 0 is the CLS token.
        B = x.shape[0]
        flat = x.view(B, -1)
        cls = self._linear(flat).unsqueeze(1)              # (B, 1, D)
        patches = torch.zeros(B, 196, self.embed_dim)      # (B, 196, D)
        return torch.cat([cls, patches], dim=1)            # (B, 197, D)


def _make_encoder(**kwargs) -> VitSceneEncoder:
    """Return a VitSceneEncoder with the fake ViT injected."""
    enc = VitSceneEncoder(model=_FakeViT(), backend="timm", **kwargs)
    enc.load()
    return enc


# ---------------------------------------------------------------------------
# Output shape & dtype
# ---------------------------------------------------------------------------


def test_forward_output_shape_batch4():
    """Output must be (B, 768) for a batch of 4 random frames."""
    enc = _make_encoder()
    x = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    out = enc(x)
    assert out.shape == (BATCH_SIZE, EMBEDDING_DIM), (
        f"Expected ({BATCH_SIZE}, {EMBEDDING_DIM}), got {out.shape}"
    )


def test_forward_output_dtype_float32():
    enc = _make_encoder()
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    assert enc(x).dtype == torch.float32


def test_forward_output_on_cpu():
    enc = _make_encoder(device="cpu")
    x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
    out = enc(x)
    assert out.device.type == "cpu"


def test_embedding_dim_matches_constant():
    """embedding_dim attribute must equal the 768 constant in the class."""
    enc = _make_encoder()
    assert enc.embedding_dim == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Frozen backbone
# ---------------------------------------------------------------------------


def test_freeze_disables_all_gradients():
    enc = _make_encoder(freeze_backbone=True)
    for param in enc.model.parameters():
        assert not param.requires_grad, "All params must be frozen after freeze()"


def test_freeze_sets_eval_mode():
    enc = _make_encoder(freeze_backbone=True)
    assert not enc.model.training, "Model must be in eval() after freeze()"


def test_unfreeze_re_enables_gradients():
    enc = _make_encoder(freeze_backbone=True)
    enc.unfreeze()
    for param in enc.model.parameters():
        assert param.requires_grad, "All params must require grad after unfreeze()"


def test_freeze_then_forward_still_runs():
    enc = _make_encoder(freeze_backbone=True)
    x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
    out = enc(x)
    assert out.shape == (2, EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# Inference latency (CPU, frozen, < 200 ms per frame)
# ---------------------------------------------------------------------------


def test_frozen_inference_latency_cpu_under_200ms():
    """Frozen backbone inference must complete in < 200 ms per frame on CPU."""
    enc = _make_encoder(freeze_backbone=True)
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    # warm-up
    enc(x)
    start = time.perf_counter()
    enc(x)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 200, f"Inference took {elapsed_ms:.1f} ms (limit 200 ms)"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_normalizes_to_imagenet_stats():
    """A black image (all zeros) after ImageNet normalisation must have negative mean.

    ImageNet mean ≈ [0.485, 0.456, 0.406]; subtracting from 0 gives negative values.
    """
    enc = _make_encoder()
    black = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    processed = enc.preprocess_frames(black)
    assert processed.mean().item() < 0, (
        "Black image after ImageNet normalisation must have negative mean (mean subtracted)"
    )


def test_preprocess_output_shape():
    enc = _make_encoder()
    raw = torch.rand(3, IMG_SIZE, IMG_SIZE)  # single HWC-like tensor
    # preprocess_frames handles 3D input by treating it as a single sample
    processed = enc.preprocess_frames(raw)
    assert processed.shape[-2:] == (IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# encode_frame / encode_batch
# ---------------------------------------------------------------------------


def test_encode_frame_returns_scene_embedding():
    enc = _make_encoder()
    frame = torch.rand(3, IMG_SIZE, IMG_SIZE)
    result = enc.encode_frame(frame, frame_id=7)
    assert isinstance(result, SceneEmbedding)
    assert result.frame_id == 7
    assert result.vector.shape == (EMBEDDING_DIM,)


def test_encode_batch_shape():
    enc = _make_encoder()
    frames = [torch.rand(3, IMG_SIZE, IMG_SIZE) for _ in range(4)]
    out = enc.encode_batch(frames)
    assert out.shape == (4, EMBEDDING_DIM)


def test_encode_batch_empty_returns_empty():
    enc = _make_encoder()
    out = enc.encode_batch([])
    assert out.shape[0] == 0
