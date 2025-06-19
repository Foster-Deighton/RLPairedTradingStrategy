#!/usr/bin/env python3
import os
import sys
import shutil
import logging
import concurrent.futures
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr

# ——— Logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("preprocess")

# ——— Parameters ———
MISSING_THRESHOLD = 0.05    # max fraction missing per ticker
WINSOR_SIGMA      = 3.0     # winsorize returns at ±3σ
MIN_RET_STD       = 1e-6    # drop near-zero-vol tickers

# ——— Paths ———
BASE_DIR        = Path(__file__).resolve().parent
RAW_DIR         = BASE_DIR / 'raw'
RAW_PRICES_DIR  = RAW_DIR / 'prices'
PROC_DIR        = BASE_DIR / 'preprocessed'

# ——— Prepare output folder ———
if PROC_DIR.exists():
    shutil.rmtree(PROC_DIR)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ——— Helper: load one ticker's adjusted-close series ———
def load_price(ticker):
    pq  = RAW_PRICES_DIR / f"{ticker}.parquet"
    csv = RAW_DIR / f"{ticker}.csv"
    df  = None

    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except Exception:
            log.warning(f"Bad parquet for {ticker}")
    elif csv.exists():
        try:
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
        except Exception:
            log.warning(f"Bad CSV for {ticker}")

    if df is None:
        return None

    # pick adjusted close if available
    if 'Adj Close' in df.columns:
        s = df['Adj Close'].copy()
    elif 'Close' in df.columns:
        s = df['Close'].copy()
    else:
        return None

    s.name = ticker
    return s

# ——— 1) Load & clean equity prices ———
parquet_tickers = [p.stem for p in RAW_PRICES_DIR.glob('*.parquet')]
csv_tickers     = [p.stem for p in RAW_DIR.glob('*.csv')]
tickers = list(set(parquet_tickers + csv_tickers))

with concurrent.futures.ThreadPoolExecutor() as exe:
    series_list = list(exe.map(load_price, tickers))

series_list = [s for s in series_list if isinstance(s, pd.Series) and not s.empty]
if not series_list:
    sys.exit("No valid price series found; aborting.")

adj = pd.concat(series_list, axis=1).sort_index()
adj = adj.asfreq('B').ffill().bfill()
min_count = int((1 - MISSING_THRESHOLD) * len(adj))
adj = adj.dropna(axis=1, thresh=min_count)
good_tickers = adj.columns.tolist()
log.info(f"{len(good_tickers)} tickers remain after missing-data filter")

# ——— 2) Fetch & save metadata (Sector + MarketCap) ———
def fetch_meta(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'Ticker':     ticker,
            'Sector':     info.get('sector', 'Unknown'),
            'MarketCap':  info.get('marketCap', np.nan)
        }
    except Exception:
        return {
            'Ticker':    ticker,
            'Sector':    'Unknown',
            'MarketCap': np.nan
        }

meta_list = []
with concurrent.futures.ThreadPoolExecutor() as exe:
    for m in exe.map(fetch_meta, good_tickers):
        meta_list.append(m)

meta_df = pd.DataFrame(meta_list)
meta_df.to_csv(PROC_DIR / 'metadata.csv', index=False)
log.info(f"Saved metadata.csv with {len(meta_df)} entries")

# ——— 3) Save cleaned prices panel ———
prices_out = PROC_DIR / 'prices.csv'
adj.to_csv(prices_out)
log.info(f"Saved prices.csv ({adj.shape[1]} tickers × {adj.shape[0]} days)")

# ——— 4) Compute, clean & save returns panel ———
rets = adj.pct_change().dropna(how='all')
sigma = rets.std()
rets = rets.clip(-WINSOR_SIGMA * sigma, WINSOR_SIGMA * sigma, axis=1)
vol = rets.std()
low_vol = vol[vol < MIN_RET_STD].index.tolist()
if low_vol:
    log.info(f"Dropping {len(low_vol)} near-zero-vol tickers")
    rets = rets.drop(columns=low_vol)
    adj  = adj.drop(columns=low_vol)
    meta_df = meta_df[~meta_df['Ticker'].isin(low_vol)]
    meta_df.to_csv(PROC_DIR / 'metadata.csv', index=False)

rets = (rets - rets.mean()) / rets.std()
returns_out = PROC_DIR / 'returns.csv'
rets.to_csv(returns_out)
log.info(f"Saved returns.csv ({rets.shape[1]} tickers × {rets.shape[0]} days)")

# ——— 5) Fetch & save macro/regime indicators ———
log.info("Fetching regime indicators...")

# a) SPY 200-day momentum
df_spy = yf.download('SPY', start=adj.index.min(), end=adj.index.max(), progress=False)['Close']
spy_mom = df_spy.pct_change(200)
spy_mom.name = 'SPY_200d_Mom'

# b) VIX
vix = yf.download('^VIX', start=adj.index.min(), end=adj.index.max(), progress=False)['Close']
vix.name = 'VIX'

# c) 10yr–2yr Treasury spread (FRED)
treas = pdr.DataReader(['DGS10','DGS2'], 'fred', adj.index.min(), adj.index.max())
ty2 = (treas['DGS10'] - treas['DGS2'])
ty2.name = '10y2y_Spread'

# d) BAA–AAA credit spread (FRED)
cred = pdr.DataReader(['BAMLC0A0CM','AAA'], 'fred', adj.index.min(), adj.index.max())
cred_spread = (cred['BAMLC0A0CM'] - cred['AAA'])
cred_spread.name = 'Credit_Spread'

regime = pd.concat([spy_mom, vix, ty2, cred_spread], axis=1)
regime = regime.asfreq('B').ffill().bfill()
regime.to_csv(PROC_DIR / 'regime.csv')
log.info("Saved regime.csv with macro/regime indicators")

# ——— 6) Save S&P 500 returns ———
log.info("Fetching S&P 500 data...")
spx = yf.download('^GSPC', start=adj.index.min(), end=adj.index.max(), progress=False)['Close']
spx_returns = spx.pct_change()
spx_returns.name = '^GSPC'
spx_returns.to_csv(PROC_DIR / 'SPX_returns.csv')
log.info("Saved SPX_returns.csv")

log.info("Preprocessing complete.")
