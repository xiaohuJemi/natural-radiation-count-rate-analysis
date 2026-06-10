from __future__ import annotations

import argparse

from analysis import main as run_old_data_analysis
from src.csv_analysis import run_csv_analysis
from src.new_data_workflow import run_new_data_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run natural gamma-ray count-rate analyses.")
    parser.add_argument(
        "--analysis",
        choices=["old", "new", "csv", "all"],
        default="old",
        help="Analysis workflow to run. Default: old.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis in {"old", "all"}:
        print("Running old unlabeled workbook analysis...")
        run_old_data_analysis()
    if args.analysis in {"new", "all"}:
        print("Running labeled caving-condition Excel analysis...")
        for path in run_new_data_analysis():
            print(path)
    if args.analysis in {"csv", "all"}:
        print("Running long CSV time-series analysis...")
        outputs = run_csv_analysis()
        print(outputs)


if __name__ == "__main__":
    main()
