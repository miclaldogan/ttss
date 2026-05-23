"""Minimal training entrypoint for TTSS experiments."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the TTSS model stack.")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to processed dataset root",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to training annotations",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("TTSS training scaffold")
    print(f"data_root={args.data_root}")
    print(f"annotations={args.annotations}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"learning_rate={args.learning_rate}")


if __name__ == "__main__":
    main()
