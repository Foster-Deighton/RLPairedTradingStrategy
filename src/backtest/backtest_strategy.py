import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ——— Logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ——— Paths ———
SCRIPT_DIR       = Path(__file__).resolve().parent           # .../rl_pairs_trading/src/backtest
SRC_DIR          = SCRIPT_DIR.parent                          # .../rl_pairs_trading/src
PAIRING_DIR      = SRC_DIR / 'pairing'
PARAMS_FILE      = PAIRING_DIR / 'pair_params.csv'
SIGNALS_DIR      = PAIRING_DIR / 'signals'
BACKTEST_OUT     = SCRIPT_DIR / 'backtest_strategy_results.csv'

# ——— Backtest settings ———
TRADING_DAYS_PER_YEAR = 252


def load_pair_params():
    df = pd.read_csv(PARAMS_FILE)
    logger.info(f"Loaded {len(df)} pair parameter entries")
    return df


def backtest_pair(a, b, params_row):
    """
    Simulate P&L for a single pair using its signals file.
    Returns a dict of metrics.
    """
    sig_file = SIGNALS_DIR / f"{a}_{b}_signals.csv"
    if not sig_file.exists():
        logger.warning(f"Signals missing for {a}-{b}, skipping")
        return None

    df = pd.read_csv(sig_file, parse_dates=['Date'])
    df = df.sort_values('Date')

    # reconstruct a daily position: hold previous signal when NaN
    df['Position'] = df['Signal'].ffill().fillna(0)

    # compute spread change (today minus yesterday)
    df['SpreadChange'] = df['Spread'].diff()

    # P&L = yesterday's position × today's spread change
    df['PnL'] = df['Position'].shift(1).fillna(0) * df['SpreadChange'].fillna(0)

    daily_pnl = df['PnL']

    # metrics
    total_pnl     = daily_pnl.sum()
    mean_daily    = daily_pnl.mean()
    std_daily     = daily_pnl.std(ddof=0)  # population std
    sharpe        = (mean_daily / std_daily) * np.sqrt(TRADING_DAYS_PER_YEAR) if std_daily else np.nan
    win_rate      = (daily_pnl > 0).mean()

    return {
        'TickerA':       a,
        'TickerB':       b,
        'PValue':        params_row['PValue'],
        'Beta':          params_row['Beta'],
        'Spread_Mean':   params_row['Spread_Mean'],
        'Spread_STD':    params_row['Spread_STD'],
        'TotalPnL':      total_pnl,
        'MeanDailyPnL':  mean_daily,
        'Sharpe':        sharpe,
        'WinRate':       win_rate
    }


def main():
    params_df = load_pair_params()
    results = []

    for _, row in params_df.iterrows():
        a, b = row['TickerA'], row['TickerB']
        metrics = backtest_pair(a, b, row)
        if metrics:
            results.append(metrics)

    if not results:
        logger.error("No backtest results—did any signals exist?")
        return

    bt_df = pd.DataFrame(results)
    bt_df.to_csv(BACKTEST_OUT, index=False)
    logger.info(f"Wrote backtest results for {len(bt_df)} pairs to {BACKTEST_OUT}")

    # Show top 10 by Sharpe
    top = bt_df.sort_values('Sharpe', ascending=False).head(10)
    print("Top 10 pairs by Sharpe ratio:")
    print(top[['TickerA','TickerB','Sharpe','TotalPnL','WinRate']])

if __name__ == '__main__':
    main()
