"""Temporal Threat Scoring System (TTSS): package evaluation CLI."""

from __future__ import annotations

import argparse

from ttss.evaluation.benchmark import BenchmarkRecord, BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    """Build the TTSS evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate TTSS benchmark runs.")
    parser.add_argument("--variant", default="ttss-baseline", help="Model variant")
    return parser


def main() -> None:
    """Run the TTSS evaluation scaffold."""
    args = build_parser().parse_args()
    runner = BenchmarkRunner()
    runner.add_record(
        BenchmarkRecord(
            variant=args.variant,
            auc=0.0,
            early_alarm_rate=0.0,
            f1=0.0,
        )
    )
    print(runner.to_table())


if __name__ == "__main__":
    main()
