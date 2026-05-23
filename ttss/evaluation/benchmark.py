"""Temporal Threat Scoring System (TTSS): benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class BenchmarkRecord:
    """Single benchmark row for a model variant."""

    variant: str
    auc: float
    early_alarm_rate: float
    f1: float


class BenchmarkRunner:
    """Collect and summarize benchmark results for TTSS experiments."""

    def __init__(self) -> None:
        self.records: list[BenchmarkRecord] = []

    def add_record(self, record: BenchmarkRecord) -> None:
        """Add a benchmark result row."""
        self.records.append(record)

    def run(self, records: Sequence[BenchmarkRecord]) -> list[BenchmarkRecord]:
        """Store and return benchmark records."""
        self.records.extend(records)
        return list(self.records)

    def to_table(self) -> list[dict[str, float | str]]:
        """Serialize the benchmark results to table-shaped dictionaries."""
        return [record.__dict__.copy() for record in self.records]
