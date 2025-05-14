#!/usr/bin/env python3
"""
Compute and summarize key performance metrics for a trained RL pairs-trading agent.

Metrics per episode:
  - Total PnL
  - Average daily return
  - Daily volatility
  - Annualized Sharpe ratio (252 trading days)
  - Win-rate (fraction of positive-return days)
  - Total number of trades
  - Maximum drawdown
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from rl.environment import PairsEnv

# Defaults relative to this script (src/evaluation/metrics.py)
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL    = HERE.parent / "rl" / "models" / "best_model.zip"
DEFAULT_EPISODES = 20


def compute_max_drawdown(cum_returns: np.ndarray) -> float:
    """
    Compute the maximum drawdown (largest peak-to-trough drop)
    of a cumulative returns series.
    """
    peak = cum_returns[0]
    max_dd = 0.0
    for x in cum_returns:
        peak = max(peak, x)
        max_dd = max(max_dd, peak - x)
    return max_dd


def main():
    parser = argparse.ArgumentParser(
        description="Compute episode-level metrics for an RL pairs-trading model"
    )
    parser.add_argument(
        "--model", "-m",
        default=str(DEFAULT_MODEL),
        help=f"Path to trained model ZIP (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--prices", "-p",
        default="../data/preprocessed/prices.csv",
        help="Path to preprocessed prices CSV"
    )
    parser.add_argument(
        "--pairs", "-c",
        default="../pairing/pair_candidates.csv",
        help="Path to selected pairs CSV"
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of episodes to simulate (default: {DEFAULT_EPISODES})"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy for evaluation (default: True)"
    )
    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load the trained agent
    model = PPO.load(str(model_path))

    # Prepare the environment
    env = PairsEnv(
        prices_csv=str(Path(__file__).parent / args.prices),
        pairs_csv=str(Path(__file__).parent / args.pairs)
    )

    records = []
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        daily_rewards = []
        prev_pos = np.zeros(len(env.pairs), dtype=int)

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # track daily reward and trades
            pos = np.where(action == 1, 1,
                           np.where(action == 2, -1, 0))
            trade_count = int(np.sum(np.abs(pos - prev_pos)))
            prev_pos = pos

            daily_rewards.append((reward, trade_count))

        # unpack
        rewards, trades = zip(*daily_rewards)
        rewards = np.array(rewards, dtype=float)
        trades  = np.array(trades, dtype=int)

        cum_returns   = np.cumsum(rewards)
        total_pnl     = cum_returns[-1]
        avg_daily     = rewards.mean()
        daily_vol     = rewards.std(ddof=1)
        annual_sharpe = (avg_daily / daily_vol) * np.sqrt(252) if daily_vol > 0 else np.nan
        win_rate      = float(np.mean(rewards > 0))
        total_trades  = int(trades.sum())
        max_dd        = compute_max_drawdown(cum_returns)

        records.append({
            "Episode":      ep,
            "TotalPnL":     total_pnl,
            "AvgDailyRet":  avg_daily,
            "DailyVol":     daily_vol,
            "AnnualSharpe": annual_sharpe,
            "WinRateDays":  win_rate,
            "TotalTrades":  total_trades,
            "MaxDrawdown":  max_dd,
        })

    # DataFrame and summary
    df = pd.DataFrame(records)
    summary = df.describe().T[["mean", "std", "min", "50%", "max"]]

    print("\nEpisode-level metrics:")
    print(df.to_string(index=False))
    print("\nSummary across episodes:")
    print(summary.to_string())

    # Save detailed CSV
    metrics_csv = HERE / "metrics_summary.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"\nDetailed metrics saved to {metrics_csv}")

if __name__ == "__main__":
    main()
