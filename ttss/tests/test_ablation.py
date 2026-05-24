"""Temporal Threat Scoring System (TTSS): tests for ablation study (issue #10).

CI check: ablation factory instantiates all variants without error.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ttss.training.ablation import (
    ARCH_VARIANTS,
    FUSION_INPUT_DIMS,
    FUSION_STRATEGIES,
    FUSION_SWEEP_CONFIGS,
    WINDOW_K_VALUES,
    WINDOW_SWEEP_CONFIGS,
    AblationConfig,
    build_ablation_loss,
    build_ablation_model,
)
from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor

B, T = 2, 16


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


def test_ablation_config_defaults() -> None:
    cfg = AblationConfig()
    assert cfg.variant == "full"
    assert cfg.use_attention is True
    assert cfg.bidirectional is True
    assert cfg.precrime_weight == pytest.approx(2.0)
    assert cfg.consistency_lambda == pytest.approx(0.1)


def test_arch_variants_has_six_keys() -> None:
    assert len(ARCH_VARIANTS) == 6
    expected = {
        "full", "no_vit", "no_attention",
        "unidirectional", "no_precrime_loss", "no_temporal_consistency",
    }
    assert set(ARCH_VARIANTS) == expected


def test_window_sweep_covers_required_k_values() -> None:
    assert WINDOW_K_VALUES == [0, 30, 60, 90, 120, 150]
    assert len(WINDOW_SWEEP_CONFIGS) == 6
    assert [c.precrime_window_k for c in WINDOW_SWEEP_CONFIGS] == WINDOW_K_VALUES


def test_fusion_strategies_defined() -> None:
    assert set(FUSION_STRATEGIES) == {"concat", "additive", "attention"}
    assert len(FUSION_SWEEP_CONFIGS) == 3


# ---------------------------------------------------------------------------
# CI check: all arch variants instantiate without error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant_name", list(ARCH_VARIANTS))
def test_build_ablation_model_instantiates(variant_name: str) -> None:
    cfg = ARCH_VARIANTS[variant_name]
    model = build_ablation_model(cfg)
    assert isinstance(model, BiLSTMThreatPredictor)


@pytest.mark.parametrize("variant_name", list(ARCH_VARIANTS))
def test_build_ablation_model_forward_pass(variant_name: str) -> None:
    cfg = ARCH_VARIANTS[variant_name]
    model = build_ablation_model(cfg)
    model.eval()
    with torch.no_grad():
        x = torch.rand(B, T, cfg.input_dim)
        pred = model(x)
    assert pred.frame_scores.shape == (B, T)
    assert float(pred.frame_scores.min()) >= 0.0
    assert float(pred.frame_scores.max()) <= 1.0


# ---------------------------------------------------------------------------
# Variant-specific structural checks
# ---------------------------------------------------------------------------


def test_no_vit_uses_small_input_dim() -> None:
    cfg = ARCH_VARIANTS["no_vit"]
    assert cfg.input_dim == 8
    model = build_ablation_model(cfg)
    assert model.input_dim == 8


def test_no_attention_disables_attention_module() -> None:
    cfg = ARCH_VARIANTS["no_attention"]
    assert cfg.use_attention is False
    model = build_ablation_model(cfg)
    assert model.attention is None


def test_unidirectional_sets_bidirectional_false() -> None:
    cfg = ARCH_VARIANTS["unidirectional"]
    assert cfg.bidirectional is False
    model = build_ablation_model(cfg)
    assert model.lstm.bidirectional is False


def test_no_precrime_loss_sets_weight_zero() -> None:
    cfg = ARCH_VARIANTS["no_precrime_loss"]
    assert cfg.precrime_weight == pytest.approx(0.0)


def test_no_temporal_consistency_sets_lambda_zero() -> None:
    cfg = ARCH_VARIANTS["no_temporal_consistency"]
    assert cfg.consistency_lambda == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fusion sweep
# ---------------------------------------------------------------------------


def test_fusion_input_dims_correct() -> None:
    assert FUSION_INPUT_DIMS["concat"] == 776
    assert FUSION_INPUT_DIMS["additive"] == 768
    assert FUSION_INPUT_DIMS["attention"] == 768


@pytest.mark.parametrize("cfg", FUSION_SWEEP_CONFIGS)
def test_fusion_model_forward_pass(cfg: AblationConfig) -> None:
    model = build_ablation_model(cfg)
    model.eval()
    with torch.no_grad():
        x = torch.rand(B, T, cfg.input_dim)
        pred = model(x)
    assert pred.frame_scores.shape == (B, T)


# ---------------------------------------------------------------------------
# build_ablation_loss
# ---------------------------------------------------------------------------


def test_build_ablation_loss_returns_callable() -> None:
    cfg = AblationConfig()
    loss_fn = build_ablation_loss(cfg)
    assert callable(loss_fn)


def test_build_ablation_loss_produces_scalar() -> None:
    cfg = AblationConfig()
    loss_fn = build_ablation_loss(cfg)
    preds = torch.rand(B, T)
    targets = torch.rand(B, T)
    loss = loss_fn(preds, targets)
    assert isinstance(loss, (float, torch.Tensor))
    assert float(loss) >= 0.0


def test_no_temporal_consistency_loss_equals_regression_only() -> None:
    from ttss.training.losses import composite_threat_loss

    cfg = AblationConfig(consistency_lambda=0.0)
    loss_fn = build_ablation_loss(cfg)
    preds = torch.rand(B, T)
    targets = torch.rand(B, T)
    ablation_loss = float(loss_fn(preds, targets))
    ref_loss = float(composite_threat_loss(preds, targets, lambda2=0.0))
    assert ablation_loss == pytest.approx(ref_loss, rel=1e-5)


# ---------------------------------------------------------------------------
# run_ablation CLI
# ---------------------------------------------------------------------------


def test_run_ablation_parser_accepts_arch() -> None:
    from ttss.scripts.run_ablation import build_parser

    args = build_parser().parse_args(["--experiment", "arch"])
    assert args.experiment == ["arch"]


def test_run_ablation_parser_accepts_all() -> None:
    from ttss.scripts.run_ablation import build_parser

    args = build_parser().parse_args(["--experiment", "all"])
    assert args.experiment == ["all"]


def test_run_ablation_arch_writes_json(tmp_path) -> None:
    import sys
    from ttss.scripts.run_ablation import main

    output_dir = tmp_path / "evaluation"
    sys.argv = [
        "run_ablation",
        "--experiment", "arch",
        "--output-dir", str(output_dir),
        "--seed", "0",
    ]
    main()
    out_file = output_dir / "ablation_arch.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["experiment"] == "arch"
    assert len(data["results"]) == 6
    for rec in data["results"]:
        assert "variant" in rec
        assert "frame_auc" in rec
        assert "early_alarm_rate" in rec


def test_run_ablation_window_writes_json(tmp_path) -> None:
    import sys
    from ttss.scripts.run_ablation import main

    output_dir = tmp_path / "evaluation"
    sys.argv = [
        "run_ablation",
        "--experiment", "window",
        "--output-dir", str(output_dir),
    ]
    main()
    out_file = output_dir / "ablation_window.json"
    data = json.loads(out_file.read_text())
    k_values = [r["K"] for r in data["results"]]
    assert k_values == WINDOW_K_VALUES


def test_run_ablation_fusion_writes_json(tmp_path) -> None:
    import sys
    from ttss.scripts.run_ablation import main

    output_dir = tmp_path / "evaluation"
    sys.argv = [
        "run_ablation",
        "--experiment", "fusion",
        "--output-dir", str(output_dir),
    ]
    main()
    out_file = output_dir / "ablation_fusion.json"
    data = json.loads(out_file.read_text())
    fusion_values = {r["fusion"] for r in data["results"]}
    assert fusion_values == set(FUSION_STRATEGIES)


def test_run_ablation_all_experiment_writes_three_files(tmp_path) -> None:
    import sys
    from ttss.scripts.run_ablation import main

    output_dir = tmp_path / "evaluation"
    sys.argv = [
        "run_ablation",
        "--experiment", "all",
        "--output-dir", str(output_dir),
    ]
    main()
    for exp in ("arch", "window", "fusion"):
        assert (output_dir / f"ablation_{exp}.json").exists()
