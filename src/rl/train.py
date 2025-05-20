#!/usr/bin/env python3
"""
Walk‐forward RL training with Sortino‐penalty, regime overlay, and rolling train/eval windows.

Run from src/rl:
    python train.py

This will:
  - Read your full prices CSV
  - Split into multiple overlapping train/eval folds based on --eval-pct
  - For each fold:
      • Slice out train & eval price CSVs
      • Instantiate PairsEnv with beta, trend_weight, sortino parameters
      • Train PPO with early stopping on eval reward
      • Save best and final models under models/fold_<i>/
  - Produce walkforward_summary.csv listing each fold's OOS mean reward
"""
import argparse
import os
import sys
from pathlib import Path
import tempfile
import pandas as pd

# Ensure environment.py is on PYTHONPATH when running "python train.py" from src/rl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList

from environment import PairsEnv

def parse_args():
    p = argparse.ArgumentParser("Walk‐forward RL training for pairs trading")
    p.add_argument("--prices",
        default="../data/preprocessed/prices.csv",
        help="Full preprocessed prices CSV (relative to src/rl)"
    )
    p.add_argument("--pairs",
        default="../pairing/pair_candidates.csv",
        help="Pair candidates CSV (relative to src/rl)"
    )
    p.add_argument("--model-dir",
        default="models",
        help="Directory under src/rl in which to save fold models"
    )
    p.add_argument("--timesteps",    type=int, default=200_000,
        help="PPO timesteps per fold"
    )
    p.add_argument("--eval-episodes", type=int, default=5,
        help="Number of episodes per evaluation"
    )
    p.add_argument("--eval-pct", type=float, default=0.2,
        help="Fraction of the data reserved for each out‐of‐sample eval window"
    )

    # Regime / market‐overlay parameters
    p.add_argument("--beta",         type=float, default=0.1,
        help="Weight on SPY return in reward (positive to penalize, negative to reward)"
    )
    p.add_argument("--trend-weight", type=float, default=0.5,
        help="Position‐sizing scale when SPY trend is strong"
    )

    # Sortino‐style penalty parameters
    p.add_argument("--sortino-window", type=int,   default=252,
        help="Lookback (days) for downside deviation"
    )
    p.add_argument("--sortino-lambda", type=float, default=5.0,
        help="Multiplier λ for downside deviation in reward"
    )

    # PPO hyperparameters
    p.add_argument("--lr",         type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma",      type=float, default=0.99, help="Discount factor")
    p.add_argument("--clip-range", type=float, default=0.2,  help="PPO clip range")
    p.add_argument("--ent-coef",   type=float, default=0.0,  help="Entropy coefficient")
    p.add_argument("--h1",         type=int,   default=64,   help="Hidden layer 1 size")
    p.add_argument("--h2",         type=int,   default=64,   help="Hidden layer 2 size")

    return p.parse_args()


def slice_fold_prices(prices_csv: str, start_idx: int, end_idx: int, dates: pd.DatetimeIndex) -> str:
    """
    Given the full dates index and start/end positions, slice the main CSV
    and write that subrange to a temp CSV, returning its path.
    """
    df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    df.index = df.index.normalize().sort_values()
    start_date = dates[start_idx]
    end_date   = dates[end_idx]
    sub = df.loc[start_date : end_date]
    if sub.empty:
        raise ValueError(f"No data in slice {start_date.date()}–{end_date.date()}")
    out = Path(tempfile.gettempdir()) / f"prices_{start_date.date()}_{end_date.date()}.csv"
    sub.to_csv(out)
    return str(out)


def main():
    args = parse_args()
    # prepare model root
    base_dir = Path(args.model_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    # load full date index
    full = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    dates = full.index.normalize().sort_values()
    n = len(dates)
    if n < 2:
        raise RuntimeError("Not enough dates in prices CSV")

    # compute train/eval window lengths
    eval_len  = max(1, int(n * args.eval_pct))
    train_len = n - eval_len
    if train_len < 1:
        raise ValueError("eval-pct too large; train window < 1")

    step = eval_len
    metrics = []

    # walk‐forward folds
    fold = 0
    for start in range(0, n - train_len - eval_len + 1, step):
        # train: [start, start+train_len-1], eval: [start+train_len, start+train_len+eval_len-1]
        ts_idx = start
        te_idx = start + train_len - 1
        es_idx = start + train_len
        ee_idx = start + train_len + eval_len - 1

        ts_date, te_date = dates[ts_idx].date(), dates[te_idx].date()
        es_date, ee_date = dates[es_idx].date(), dates[ee_idx].date()
        print(f"\n=== Fold {fold}: TRAIN {ts_date}→{te_date}, EVAL {es_date}→{ee_date} ===")

        # slice out train & eval CSVs
        train_csv = slice_fold_prices(args.prices, ts_idx, te_idx, dates)
        eval_csv  = slice_fold_prices(args.prices, es_idx, ee_idx, dates)

        # fold directory
        fold_dir = base_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True, parents=True)

        # environments
        train_env = DummyVecEnv([lambda: Monitor(
            PairsEnv(
                prices_csv=train_csv,
                pairs_csv=args.pairs,
                window_length=90,
                trading_cost=0.0005
            ),
            filename=str(fold_dir / "train_monitor.csv")
        )])
        eval_env = DummyVecEnv([lambda: Monitor(
            PairsEnv(
                prices_csv=eval_csv,
                pairs_csv=args.pairs,
                window_length=90,
                trading_cost=0.0005
            ),
            filename=str(fold_dir / "eval_monitor.csv")
        )])

        # early stopping callback
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3,
            min_evals=5,
            verbose=1
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(fold_dir),
            log_path=str(fold_dir),
            eval_freq=10_000,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            callback_after_eval=stop_cb
        )

        # instantiate PPO
        policy_kwargs = dict(net_arch=[args.h1, args.h2])
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            gamma=args.gamma,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(fold_dir / "tb_logs")
        )

        # train
        model.learn(total_timesteps=args.timesteps, callback=CallbackList([eval_cb]))
        model.save(str(fold_dir / "final_model"))

        # record OOS performance
        eval_df = pd.read_csv(fold_dir / "eval_monitor.csv", comment="#")
        mean_oos = eval_df["r"].mean()
        print(f"Fold {fold} mean OOS reward: {mean_oos:.4f}")
        metrics.append({
            "fold": fold,
            "train_start": ts_date,
            "train_end":   te_date,
            "eval_start":  es_date,
            "eval_end":    ee_date,
            "mean_oos_reward": mean_oos
        })

        fold += 1

    # summary
    summary = pd.DataFrame(metrics)
    summary["overall_mean_oos"] = summary["mean_oos_reward"].mean()
    summary.to_csv(base_dir / "walkforward_summary.csv", index=False)
    print("\n=== Walk‐forward summary ===")
    print(summary)

if __name__ == "__main__":
    main()
