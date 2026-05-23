"""Unit tests for the BiLSTM prediction layer (issue #5).

Tests use random tensors — no pre-trained weights needed.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor, ThreatPrediction

B = 2
T = 16
FEATURE_DIM = 1536  # fused YOLO-8 + ViT-768 × 2


def _make_predictor(**kwargs) -> BiLSTMThreatPredictor:
    defaults = dict(input_dim=FEATURE_DIM)
    defaults.update(kwargs)
    return BiLSTMThreatPredictor(**defaults)


# ---------------------------------------------------------------------------
# Forward pass shape & dtype
# ---------------------------------------------------------------------------


def test_forward_output_type():
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    result = pred(x)
    assert isinstance(result, ThreatPrediction)


def test_forward_frame_scores_shape():
    """frame_scores must be (B, T)."""
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    result = pred(x)
    assert result.frame_scores.shape == (B, T), (
        f"Expected ({B}, {T}), got {result.frame_scores.shape}"
    )


def test_forward_sequence_score_shape():
    """sequence_score must be (B,)."""
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    result = pred(x)
    assert result.sequence_score.shape == (B,)


def test_forward_attention_weights_shape():
    """attention_weights must be (B, T)."""
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    result = pred(x)
    assert result.attention_weights.shape == (B, T)


# ---------------------------------------------------------------------------
# Output value constraints
# ---------------------------------------------------------------------------


def test_frame_scores_in_zero_one():
    """All per-frame scores must be strictly in [0, 1] (Sigmoid applied)."""
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    result = pred(x)
    assert result.frame_scores.min().item() >= 0.0
    assert result.frame_scores.max().item() <= 1.0


def test_sequence_score_in_zero_one():
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    result = pred(x)
    for s in result.sequence_score.tolist():
        assert 0.0 <= s <= 1.0, f"Sequence score {s} out of [0, 1]"


def test_attention_weights_sum_to_one():
    """Attention weights (softmax) must sum to 1.0 per sequence."""
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    result = pred(x)
    row_sums = result.attention_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), (
        f"Attention row sums: {row_sums.tolist()}"
    )


# ---------------------------------------------------------------------------
# Optional return_attention second output
# ---------------------------------------------------------------------------


def test_return_attention_flag_returns_tuple():
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    out = pred(x, return_attention=True)
    assert isinstance(out, tuple) and len(out) == 2


def test_return_attention_scores_shape():
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    scores, attn = pred(x, return_attention=True)
    assert scores.shape == (B, T)
    assert attn.shape == (B, T)


def test_return_attention_matches_threat_prediction():
    """return_attention=True values must match ThreatPrediction fields."""
    pred = _make_predictor()
    pred.eval()  # disable dropout so both forward passes are deterministic
    x = torch.rand(B, T, FEATURE_DIM)
    with torch.no_grad():
        scores, attn = pred(x, return_attention=True)
        result = pred(x)
    assert torch.allclose(scores, result.frame_scores, atol=1e-6)
    assert torch.allclose(attn, result.attention_weights, atol=1e-6)


# ---------------------------------------------------------------------------
# 2-D input (single sequence, no batch dim)
# ---------------------------------------------------------------------------


def test_2d_input_is_accepted():
    pred = _make_predictor()
    x = torch.rand(T, FEATURE_DIM)  # (T, F)
    result = pred(x)
    assert result.frame_scores.shape == (1, T)


# ---------------------------------------------------------------------------
# Parameter count logging
# ---------------------------------------------------------------------------


def test_param_count_logged_on_init(caplog):
    with caplog.at_level(logging.INFO, logger="ttss.models.prediction.bilstm_threat"):
        _ = _make_predictor()
    assert any("params=" in record.message for record in caplog.records), (
        "Expected 'params=...' in INFO log on init"
    )


def test_param_count_is_positive():
    pred = _make_predictor()
    n = sum(p.numel() for p in pred.parameters())
    assert n > 0


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


def test_config_has_required_keys():
    import yaml
    import pathlib
    # __file__ = ttss/tests/test_prediction.py  →  parents[1] = ttss/
    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "prediction.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    model_cfg = cfg["model"]
    for key in ("hidden_size", "num_layers", "dropout", "seq_len"):
        assert key in model_cfg, f"Missing key '{key}' in configs/prediction.yaml"


# ---------------------------------------------------------------------------
# Mask support
# ---------------------------------------------------------------------------


def test_padding_mask_accepted():
    pred = _make_predictor()
    x = torch.rand(B, T, FEATURE_DIM)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -4:] = False  # last 4 steps of first sequence are padding
    result = pred(x, mask=mask)
    assert result.frame_scores.shape == (B, T)
