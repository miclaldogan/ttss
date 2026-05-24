"""Temporal Threat Scoring System (TTSS): tests for baseline comparison suite."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ttss.baselines import REGISTRY, BaselinePredictor
from ttss.baselines.mean_feature_svm import MeanFeatureSVMBaseline
from ttss.baselines.rtfm import RTFMBaseline
from ttss.baselines.sultani2018 import Sultani2018Baseline


# ---------------------------------------------------------------------------
# Protocol / registry
# ---------------------------------------------------------------------------


def test_registry_contains_all_three_baselines() -> None:
    assert set(REGISTRY) == {"sultani2018", "rtfm", "mean_feature_svm"}


def test_all_baselines_satisfy_protocol() -> None:
    for name, cls in REGISTRY.items():
        instance = cls()
        assert isinstance(instance, BaselinePredictor), f"{name} does not satisfy protocol"


# ---------------------------------------------------------------------------
# Sultani 2018
# ---------------------------------------------------------------------------


def test_sultani2018_name() -> None:
    assert Sultani2018Baseline.name == "sultani2018"


def test_sultani2018_predict_video_returns_float32_array() -> None:
    pred = Sultani2018Baseline(n_frames=32)
    scores = pred.predict_video("synthetic://test_video_0")
    assert scores.dtype == np.float32
    assert scores.ndim == 1
    assert len(scores) == 32


def test_sultani2018_scores_in_unit_interval() -> None:
    pred = Sultani2018Baseline(n_frames=64)
    scores = pred.predict_video("synthetic://test_video_1")
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_sultani2018_missing_checkpoint_warns() -> None:
    with pytest.warns(UserWarning, match="checkpoint not found"):
        pred = Sultani2018Baseline(checkpoint_path="nonexistent.pt", n_frames=16)
    scores = pred.predict_video("synthetic://v")
    assert len(scores) == 16


def test_sultani2018_deterministic_for_same_video_id() -> None:
    pred = Sultani2018Baseline(n_frames=32)
    s1 = pred.predict_video("synthetic://same_video")
    s2 = pred.predict_video("synthetic://same_video")
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# RTFM
# ---------------------------------------------------------------------------


def test_rtfm_name() -> None:
    assert RTFMBaseline.name == "rtfm"


def test_rtfm_predict_video_returns_float32_array() -> None:
    pred = RTFMBaseline(n_frames=32)
    scores = pred.predict_video("synthetic://test_video_0")
    assert scores.dtype == np.float32
    assert scores.ndim == 1
    assert len(scores) == 32


def test_rtfm_scores_in_unit_interval() -> None:
    pred = RTFMBaseline(n_frames=64)
    scores = pred.predict_video("synthetic://test_video_2")
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_rtfm_missing_checkpoint_warns() -> None:
    with pytest.warns(UserWarning, match="checkpoint not found"):
        pred = RTFMBaseline(checkpoint_path="nonexistent.pt", n_frames=16)
    scores = pred.predict_video("synthetic://v")
    assert len(scores) == 16


def test_rtfm_deterministic_for_same_video_id() -> None:
    pred = RTFMBaseline(n_frames=32)
    s1 = pred.predict_video("synthetic://same_video")
    s2 = pred.predict_video("synthetic://same_video")
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# MeanFeatureSVMBaseline
# ---------------------------------------------------------------------------


def test_mean_feature_svm_name() -> None:
    assert MeanFeatureSVMBaseline.name == "mean_feature_svm"


def test_mean_feature_svm_predict_video_returns_float32_array() -> None:
    pred = MeanFeatureSVMBaseline(n_frames=32)
    scores = pred.predict_video("synthetic://test_video_0")
    assert scores.dtype == np.float32
    assert scores.ndim == 1
    assert len(scores) == 32


def test_mean_feature_svm_scores_in_unit_interval() -> None:
    pred = MeanFeatureSVMBaseline(n_frames=64)
    scores = pred.predict_video("synthetic://test_video_3")
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_mean_feature_svm_missing_checkpoint_warns() -> None:
    with pytest.warns(UserWarning, match="checkpoint not found"):
        pred = MeanFeatureSVMBaseline(checkpoint_path="nonexistent.pkl", n_frames=16)
    scores = pred.predict_video("synthetic://v")
    assert len(scores) == 16


def test_mean_feature_svm_deterministic_for_same_video_id() -> None:
    pred = MeanFeatureSVMBaseline(n_frames=32)
    s1 = pred.predict_video("synthetic://same_video")
    s2 = pred.predict_video("synthetic://same_video")
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# evaluate_baselines CLI helpers
# ---------------------------------------------------------------------------


def test_evaluate_baselines_parser_accepts_all() -> None:
    from ttss.scripts.evaluate_baselines import build_parser

    args = build_parser().parse_args(["--baseline", "all", "--split", "test"])
    assert args.baseline == ["all"]
    assert args.split == "test"


def test_evaluate_baselines_resolve_all() -> None:
    from ttss.scripts.evaluate_baselines import _resolve_baselines

    names = _resolve_baselines(["all"])
    assert set(names) == {"sultani2018", "rtfm", "mean_feature_svm"}


def test_evaluate_baselines_resolve_single() -> None:
    from ttss.scripts.evaluate_baselines import _resolve_baselines

    names = _resolve_baselines(["rtfm"])
    assert names == ["rtfm"]


def test_evaluate_baselines_unknown_raises() -> None:
    from ttss.scripts.evaluate_baselines import _resolve_baselines

    with pytest.raises(ValueError, match="Unknown baseline"):
        _resolve_baselines(["ghost_model"])


def test_evaluate_baseline_returns_correct_schema() -> None:
    from ttss.scripts.evaluate_baselines import _evaluate_baseline, _synthetic_labels

    rng = np.random.default_rng(0)
    videos = _synthetic_labels(3, 32, rng)
    records = _evaluate_baseline("sultani2018", videos, threshold=0.5, n_frames=32)
    assert len(records) == 3
    for rec in records:
        assert set(rec) == {"baseline", "video_id", "frame_auc", "early_alarm_rate"}
        assert rec["baseline"] == "sultani2018"
        assert isinstance(rec["frame_auc"], float)
        assert isinstance(rec["early_alarm_rate"], float)


def test_evaluate_baselines_main_writes_json(tmp_path) -> None:
    import sys
    from ttss.scripts.evaluate_baselines import main

    output_file = tmp_path / "baseline_results.json"
    sys.argv = [
        "evaluate_baselines",
        "--baseline", "all",
        "--split", "test",
        "--n-videos", "3",
        "--n-frames", "32",
        "--output", str(output_file),
    ]
    main()
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert "results" in data
    assert len(data["results"]) == 9  # 3 baselines × 3 videos
    required_keys = {"baseline", "video_id", "frame_auc", "early_alarm_rate"}
    for rec in data["results"]:
        assert required_keys <= set(rec)
