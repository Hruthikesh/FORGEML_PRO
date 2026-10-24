"""Run from repository root: python -m scripts.train [--data-dir data/raw]."""

import argparse
from pathlib import Path

from forgeml.training import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    report = run_training(args.output_dir, args.data_dir, args.rows, optuna_trials=args.trials)
    print(report["mode"])
    print("Final holdout:", report["holdout"])
