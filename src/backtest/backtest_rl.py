#!/usr/bin/env python3
"""
Backtest a frozen RL pairs‐trading policy over the full historical span,
and compare its actual percent returns (PnL/exposure) against the SPY benchmark.

Usage (from src/backtest/):
    python backtest_rl.py

Outputs:
  - backtest_results.csv        (Date, DailyPnL, EquityCurve, Exposure, DailyRet_RL, CumRet_RL, DailyRet_SPY, CumRet_SPY)
  - backtest_equity.png         (RL cumulative PnL)
  - backtest_returns_vs_spy.png (RL % return vs. SPY % return)
"""

import argparse
from pathlib import Path
import sys, os

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO
from rl.environment import PairsEnv

HERE = Path(__file__).resolve().parent

def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest RL policy and compute returns as PnL/exposure"
    )
    p.add_argument("-m","--model",
        default="../rl/models/best_model.zip",
        help="Path to trained model ZIP"
    )
    p.add_argument("-p","--prices",
        default="../data/preprocessed/prices.csv",
        help="Path to preprocessed prices CSV"
    )
    p.add_argument("-c","--pairs",
        default="../pairing/pair_candidates.csv",
        help="Path to selected pairs CSV"
    )
    p.add_argument("-o","--output-csv",
        default="backtest_results.csv",
        help="Filename for output CSV"
    )
    p.add_argument("-e","--equity-plot",
        default="backtest_equity.png",
        help="Filename for RL cumulative PnL plot"
    )
    p.add_argument("-r","--returns-plot",
        default="backtest_returns_vs_spy.png",
        help="Filename for RL % return vs SPY % return plot"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # resolve paths
    model_path = (HERE/args.model).resolve()
    prices_csv = (HERE/args.prices).resolve()
    pairs_csv  = (HERE/args.pairs).resolve()
    out_csv    = HERE/args.output_csv
    eq_plot    = HERE/args.equity_plot
    ret_plot   = HERE/args.returns_plot

    # validate files
    for fp,desc in [(model_path,"Model"),(prices_csv,"Prices CSV"),(pairs_csv,"Pairs CSV")]:
        if not fp.is_file():
            raise FileNotFoundError(f"{desc} not found: {fp}")

    # load price DataFrame for exposure calculation
    prices_df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    # load pairs list
    pairs_df = pd.read_csv(pairs_csv)
    pair_list = list(zip(pairs_df['TickerA'], pairs_df['TickerB']))

    # load RL policy
    model = PPO.load(str(model_path))

    # build env
    env = PairsEnv(prices_csv=str(prices_csv), pairs_csv=str(pairs_csv))

    # start at first valid date
    env.t = env.window
    env.prev_pos = np.zeros(len(env.pairs), dtype=int)
    obs = env._get_obs()

    # backtest loop
    done = False
    dates, daily_pnls, equity, exposures = [], [], [0.0], []
    while not done:
        # current date
        cur_date = env.dates[env.t]
        # compute exposure = sum(|pos| * (priceA + priceB))
        pos = env.prev_pos  # positions held at date
        prices_today = prices_df.loc[cur_date]
        exposure = 0.0
        for i, (a,b) in enumerate(pair_list):
            if pos[i] != 0:
                exposure += abs(pos[i]) * (prices_today[a] + prices_today[b])
        exposures.append(exposure)

        # step RL
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # record
        daily_pnls.append(reward)
        equity.append(equity[-1] + reward)
        dates.append(cur_date)

    # assemble DataFrame
    df = pd.DataFrame({
        "Date"        : pd.to_datetime(dates),
        "DailyPnL"    : daily_pnls,
        "EquityCurve" : equity[1:],
        "Exposure"    : exposures
    })
    # percent returns RL = PnL / exposure
    df["DailyRet_RL"] = df["DailyPnL"] / df["Exposure"].replace(0, np.nan)
    df["DailyRet_RL"].fillna(0, inplace=True)
    df["CumRet_RL"] = (1 + df["DailyRet_RL"]).cumprod() - 1

    # save
    df.to_csv(out_csv, index=False)
    print(f"Backtest CSV saved to {out_csv}")

    # plot RL cumulative PnL
    plt.figure(figsize=(10,5))
    plt.plot(df["Date"], df["EquityCurve"],
             label="RL Cumulative PnL", color="tab:blue")
    plt.xlabel("Date"); plt.ylabel("Cumulative PnL")
    plt.title("RL Policy Backtest Equity Curve")
    plt.tight_layout(); plt.savefig(eq_plot); plt.close()
    print(f"Equity curve plot saved to {eq_plot}")

    # fetch SPY & compute its percent returns
    df.set_index("Date", inplace=True)
    start = df.index[0].strftime("%Y-%m-%d")
    end   = (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_hist = yf.Ticker("SPY").history(start=start, end=end)
    if spy_hist.index.tz is not None:
        spy_hist.index = spy_hist.index.tz_localize(None)
    col = "Adj Close" if "Adj Close" in spy_hist.columns else "Close"
    spy = spy_hist[col].reindex(df.index).ffill().bfill()
    df["DailyRet_SPY"] = spy.pct_change().fillna(0)
    df["CumRet_SPY"]   = (1 + df["DailyRet_SPY"]).cumprod() - 1

    # plot percent-return overlay
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["CumRet_RL"],
             label="RL % Return (PnL/Exposure)", color="tab:blue")
    plt.plot(df.index, df["CumRet_SPY"],
             label="SPY % Return", color="tab:orange")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.title("RL Strategy vs. SPY Benchmark (Percent Returns)")
    plt.legend(); plt.tight_layout(); plt.savefig(ret_plot); plt.close()
    print(f"Percent-returns comparison plot saved to {ret_plot}")

    # print summary
    final = df["CumRet_RL"].iloc[-1]
    daily = df["DailyRet_RL"] * 100
    print("\nRL Daily Return Summary (%):")
    print(f"  Mean   : {daily.mean():.4f}%")
    print(f"  Median : {daily.median():.4f}%")
    print(f"  Min    : {daily.min():.4f}%")
    print(f"  Max    : {daily.max():.4f}%")
    print(f"\nFinal Cumulative Return: {final*100:.2f}%")

if __name__ == "__main__":
    main()
