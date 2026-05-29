"""Unit tests for PerClassEvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from ttss.evaluation.per_class import PerClassEvaluator, PerClassResult
from ttss.evaluation.temporal_eval import UCF_CRIME_CATEGORIES


class _FakeAnn:
    def __init__(self, label: str, T: int = 200, onset: int = 150):
        self.label = label
        self.y_true = np.zeros(T)
        self.y_true[onset:] = 1


def _make_scores(rng, T=200, onset=150, signal=0.4):
    base = rng.random(T)
    base[onset - 20:] += signal
    return np.clip(base, 0, 1)


def test_all_13_categories_present():
    """evaluate() returns a row for every UCF-Crime category supplied."""
    rng = np.random.default_rng(0)
    preds, anns = [], []
    for cat in UCF_CRIME_CATEGORIES:
        for _ in range(5):
            preds.append(_make_scores(rng))
            anns.append(_FakeAnn(cat))

    evaluator = PerClassEvaluator()
    results = evaluator.evaluate(preds, anns)
    categories = {r.category for r in results}

    for cat in UCF_CRIME_CATEGORIES:
        assert cat in categories, f"Missing category: {cat}"


def test_macro_row_appended():
    """evaluate() appends a '_macro' row as the final entry."""
    rng = np.random.default_rng(1)
    preds = [_make_scores(rng) for _ in range(10)]
    anns = [_FakeAnn("Abuse") for _ in range(5)] + [_FakeAnn("Robbery") for _ in range(5)]

    results = PerClassEvaluator().evaluate(preds, anns)
    assert results[-1].category == "_macro"


def test_macro_auc_weighted_by_n_videos():
    """Macro AUC is weighted by video count, not a simple average."""
    rng = np.random.default_rng(2)

    # Category A: 10 videos, high signal → high AUC
    # Category B: 2 videos, low signal → low AUC
    preds_a = [_make_scores(rng, signal=0.6) for _ in range(10)]
    preds_b = [_make_scores(rng, signal=0.05) for _ in range(2)]
    anns_a = [_FakeAnn("Assault") for _ in range(10)]
    anns_b = [_FakeAnn("Shoplifting") for _ in range(2)]

    results = PerClassEvaluator().evaluate(preds_a + preds_b, anns_a + anns_b)
    macro = next(r for r in results if r.category == "_macro")
    assault = next(r for r in results if r.category == "Assault")
    shoplifting = next(r for r in results if r.category == "Shoplifting")

    # Weighted macro should be closer to assault AUC (10 videos vs 2)
    expected = (assault.frame_auc * 10 + shoplifting.frame_auc * 2) / 12
    assert abs(macro.frame_auc - expected) < 1e-6, "Macro AUC not correctly weighted"


def test_sorted_by_auc_descending():
    """Results (excluding _macro) should be sorted by frame_auc descending."""
    rng = np.random.default_rng(3)
    preds, anns = [], []
    for cat in ["Abuse", "Robbery", "Shooting", "Vandalism"]:
        for _ in range(8):
            preds.append(_make_scores(rng))
            anns.append(_FakeAnn(cat))

    results = PerClassEvaluator().evaluate(preds, anns)
    non_macro = [r for r in results if r.category != "_macro"]
    aucs = [r.frame_auc for r in non_macro]
    assert aucs == sorted(aucs, reverse=True), "Results not sorted by AUC descending"


def test_per_class_result_to_dict_schema():
    """PerClassResult.to_dict() matches the required JSON schema."""
    r = PerClassResult(category="Abuse", n_videos=10, frame_auc=0.82,
                       ear=0.55, malt_frames=37.2, precrime_ap=0.61)
    d = r.to_dict()
    required_keys = {"category", "n_videos", "frame_auc", "ear", "malt_frames"}
    assert required_keys.issubset(set(d.keys())), f"Missing keys: {required_keys - set(d.keys())}"
    assert isinstance(d["category"], str)
    assert isinstance(d["n_videos"], int)
    assert isinstance(d["frame_auc"], float)
