#!/usr/bin/env python3
"""
Backtest a frozen RL pairs‐trading policy over the full historical span,
and compare it against the SPY benchmark.

Usage (from src/backtest/):
    python backtest_rl.py

Outputs:
  - backtest_results.csv        (Date, DailyPnL, EquityCurve)
  - backtest_equity.png         (RL cumulative PnL)
  - backtest_equity_vs_spy.png  (RL vs. SPY comparison, twin axes)
"""

import argparse
from pathlib import Path
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Silence OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO
from rl.environment import PairsEnv

# Directory this script lives in (src/backtest/)
HERE = Path(__file__).resolve().parent

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a full‐history backtest of a trained RL policy and benchmark it"
    )
    p.add_argument("--model", "-m",
        default="../rl/models/best_model.zip",
        help="Path to trained model ZIP"
    )
    p.add_argument("--prices", "-p",
        default="../data/preprocessed/prices.csv",
        help="Path to preprocessed prices CSV"
    )
    p.add_argument("--pairs", "-c",
        default="../pairing/pair_candidates.csv",
        help="Path to selected pairs CSV"
    )
    p.add_argument("--output-csv", "-o",
        default="backtest_results.csv",
        help="Filename for output CSV"
    )
    p.add_argument("--output-plot", "-q",
        default="backtest_equity.png",
        help="Filename for RL equity plot"
    )
    p.add_argument("--benchmark-plot", "-b",
        default="backtest_equity_vs_spy.png",
        help="Filename for RL vs SPY plot"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Resolve files
    model_path = (HERE / args.model).resolve()
    prices_csv = (HERE / args.prices).resolve()
    pairs_csv  = (HERE / args.pairs).resolve()
    out_csv    = HERE / args.output_csv
    out_plot   = HERE / args.output_plot
    bench_plot = HERE / args.benchmark_plot

    # Validate existence
    for fp, desc in [(model_path, "Model"), (prices_csv, "Prices CSV"), (pairs_csv, "Pairs CSV")]:
        if not fp.is_file():
            raise FileNotFoundError(f"{desc} not found: {fp}")

    # 1) Load the trained RL policy
    model = PPO.load(str(model_path))

    # 2) Build the environment
    env = PairsEnv(prices_csv=str(prices_csv), pairs_csv=str(pairs_csv))

    # 3) Start at the first valid date
    env.t = env.window
    env.prev_pos = np.zeros(len(env.pairs), dtype=int)
    obs = env._get_obs()

    # 4) Roll through the full history
    done = False
    dates, daily_pnls, equity = [], [], [0.0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        daily_pnls.append(reward)
        equity.append(equity[-1] + reward)
        dates.append(env.dates[env.t])

    # 5) Save RL backtest results
    df = pd.DataFrame({
        "Date":        dates,
        "DailyPnL":    daily_pnls,
        "EquityCurve": equity[1:]  # drop initial zero
    })
    df.to_csv(out_csv, index=False)
    print(f"Backtest CSV saved to {out_csv}")

    # 6) Plot RL equity curve
    plt.figure(figsize=(10,5))
    plt.plot(df["Date"], df["EquityCurve"], label="RL Strategy", color="tab:blue")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.title("RL Policy Backtest Equity Curve")
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Equity curve plot saved to {out_plot}")

    # --- 7) Benchmark: Fetch & plot SPY alongside on twin axes ---
    try:
        rl_index = pd.to_datetime(dates)
        start = rl_index[0].strftime("%Y-%m-%d")
        end   = (rl_index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Fetch with yfinance
        spy_hist = yf.Ticker("SPY").history(start=start, end=end)
        if spy_hist.empty:
            raise ValueError("No SPY data returned")

        # Drop timezone info if present
        if spy_hist.index.tz is not None:
            spy_hist.index = spy_hist.index.tz_localize(None)

        # Choose price column
        price_col = "Adj Close" if "Adj Close" in spy_hist.columns else "Close"
        spy_series = spy_hist[price_col].copy()

        # Align to RL dates
        spy_series = spy_series.reindex(rl_index).ffill().bfill()
        if spy_series.isna().all():
            raise ValueError("Aligned SPY series is all NaN")

        # Compute SPY cumulative return (starting from 1)
        spy_ret    = spy_series.pct_change().fillna(0)
        spy_equity = (1 + spy_ret).cumprod()

        # Build RL equity series
        rl_equity = pd.Series(equity[1:], index=rl_index, name="RL")

        # Plot both on twin axes
        fig, ax1 = plt.subplots(figsize=(10,5))
        ax1.plot(rl_equity.index, rl_equity.values,
                 label="RL Cumulative PnL", color="tab:blue")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("RL Cumulative PnL", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(spy_equity.index, spy_equity.values - 1,
                 label="SPY Cumulative Return", color="tab:orange")
        ax2.set_ylabel("SPY Cumulative Return", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("RL Strategy vs. SPY Benchmark")
        fig.tight_layout()
        fig.savefig(bench_plot)
        plt.close(fig)
        print(f"Benchmark comparison plot saved to {bench_plot}")

    except Exception as e:
        print(f"Warning: could not fetch or plot SPY benchmark ({e})")

if __name__ == "__main__":
    main()
