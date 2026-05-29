"""Temporal Threat Scoring System (TTSS): evaluation layer exports."""

from ttss.evaluation.benchmark import BenchmarkRecord, BenchmarkRunner
from ttss.evaluation.temporal_eval import (
    CategoryReport,
    EvaluationReport,
    TemporalEvaluationResult,
    TemporalEvaluator,
    UCF_CRIME_CATEGORIES,
    evaluate_sequence,
    per_category_metrics,
)

__all__ = [
    "BenchmarkRecord",
    "BenchmarkRunner",
    "CategoryReport",
    "EvaluationReport",
    "TemporalEvaluationResult",
    "TemporalEvaluator",
    "UCF_CRIME_CATEGORIES",
    "evaluate_sequence",
    "per_category_metrics",
]
