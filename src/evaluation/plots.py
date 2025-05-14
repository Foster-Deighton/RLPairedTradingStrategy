#!/usr/bin/env python3
"""
Generate learning and evaluation curves from RL training artifacts.

When run from src/evaluation/, defaults will point at:
  - ../rl/models            (for --models)
  - ./plots                 (for --output)
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Location of this script: src/evaluation
HERE = Path(__file__).resolve().parent
# Default model directories: ../rl/models
DEFAULT_MODELS   = HERE.parent / "rl" / "models"
# Default output directory: ./plots
DEFAULT_OUTPUT   = HERE / "plots"


def plot_training_curve(model_dirs, output_dir):
    plt.figure()
    for md in model_dirs:
        train_csv = Path(md) / "train_monitor.csv"
        if not train_csv.exists():
            print(f"[warn] {train_csv} not found, skipping")
            continue
        # Skip the comment line and read the CSV
        df = pd.read_csv(train_csv, comment='#')
        plt.plot(df["t"], df["r"], label=Path(md).name)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png")
    plt.close()

def plot_evaluation_curve(model_dirs, output_dir):
    plt.figure()
    for md in model_dirs:
        npz = Path(md) / "evaluations.npz"
        if not npz.exists():
            print(f"[warn] {npz} not found, skipping")
            continue
        data = np.load(npz)
        ts = data["timesteps"]
        res = data["results"]
        if res.ndim > 1:
            res = res.mean(axis=1)
        plt.plot(ts, res, label=Path(md).name)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Eval Reward")
    plt.title("Evaluation Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eval_curve.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot RL training and evaluation curves"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=[str(DEFAULT_MODELS)],
        help=f"One or more model dirs (default: {DEFAULT_MODELS})"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help=f"Directory to save plots (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curve(args.models, out_dir)
    plot_evaluation_curve(args.models, out_dir)

    print(f"Plots saved to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
