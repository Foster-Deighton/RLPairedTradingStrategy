#!/usr/bin/env python3
"""
fetch_data.py

Builds and incrementally refreshes a price panel for a broad universe:
- ALL U.S. equities (Nasdaq‐ & NYSE‐listed) via NasdaqTrader symbol files
- Plus S&P 500, NASDAQ-100, Russell-600 constituents from Wikipedia
- Plus sector ETFs
- Market-cap & liquidity filters
- Gap-fill, sliding window, parquet storage
- Fully configurable via CLI
"""

import os
import time
import logging
import argparse
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests
import yfinance as yf
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("fetch_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def fetch_wiki_tickers(url: str, column: str) -> pd.Series:
    try:
        tables = pd.read_html(url)
    except Exception as e:
        logger.warning(f"Unable to fetch tickers from {url}: {e}")
        return pd.Series(dtype=str)
    for tbl in tables:
        if column in tbl.columns:
            return tbl[column].astype(str).str.replace(r'\.A$', '.A', regex=True).dropna().unique()
    return pd.Series(dtype=str)

def fetch_nasdaq_listed() -> pd.Series:
    """Fetch all symbols from NasdaqTrader’s nasdaqlisted.txt & otherlisted.txt."""
    base = "http://ftp.nasdaqtrader.com/dynamic/SymDir"
    files = ["nasdaqlisted.txt", "otherlisted.txt"]
    syms = set()
    for fn in files:
        url = f"{base}/{fn}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            lines = r.text.splitlines()[1:-1]  # skip header/trailer
            for line in lines:
                parts = line.split("|")
                sym = parts[0].strip()
                if sym and sym not in ("Symbol",):
                    syms.add(sym)
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
    logger.info(f"Fetched {len(syms)} symbols from NasdaqTrader files")
    return pd.Series(list(syms))

def build_universe(sources: list, etfs: list, max_tickers: int, min_cap: float, jobs: int) -> pd.DataFrame:
    """
    1) Scrape Wikipedia indices
    2) Add sector ETFs
    3) Add ALL US-listed via NasdaqTrader
    4) Fetch market caps in parallel
    5) Filter by min_cap, take top N (or all if max_tickers ≤ 0)
    Returns DataFrame [Ticker, AssetType, MarketCap]
    """
    # 1) Wiki indices
    eq_syms = pd.Series(dtype=str)
    for url,col in sources:
        syms = fetch_wiki_tickers(url, col)
        logger.info(f"Scraped {len(syms)} tickers from {url}")
        eq_syms = pd.concat([eq_syms, pd.Series(syms)], ignore_index=True)
    eq_syms = set(eq_syms.dropna().tolist())

    # 2) sector ETFs
    etf_set = set(etfs)

    # 3) all US-listed
    all_listed = set(fetch_nasdaq_listed())

    # combine
    universe = eq_syms | etf_set | all_listed
    logger.info(f"Combined universe before filtering: {len(universe)} symbols")

    # 4) fetch market caps
    def _get_cap(sym):
        try:
            info = yf.Ticker(sym).info
            return sym, info.get('marketCap') or 0
        except Exception:
            return sym, 0

    cap_list = Parallel(n_jobs=jobs)(
        delayed(_get_cap)(sym) for sym in tqdm(universe, desc="Fetching marketCaps", unit="sym")
    )
    df = pd.DataFrame(cap_list, columns=['Ticker','MarketCap'])
    # tag asset type
    df['AssetType'] = df['Ticker'].apply(lambda t: 'ETF' if t in etf_set else 'Equity')
    # 5) apply cap filter & sort
    df = df[df['MarketCap'] >= min_cap]
    if max_tickers > 0 and len(df) > max_tickers:
        df = df.nlargest(max_tickers, 'MarketCap')
    df = df.reset_index(drop=True)
    logger.info(f"Filtered to {len(df)} symbols with cap ≥ {min_cap}")
    return df

def download_chunk(chunk, start, end, retries, backoff):
    for attempt in range(1, retries+1):
        try:
            return yf.download(
                chunk,
                start=start, end=end,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )
        except Exception as e:
            delay = backoff ** attempt
            logger.warning(f"Chunk {chunk[:3]}… attempt {attempt} failed: {e}, retry in {delay}s")
            time.sleep(delay)
    logger.error(f"Chunk {chunk[:3]} failed after {retries} retries")
    return pd.DataFrame()

def clean_and_filter(df_chunk, calendar, min_vol, missing, window):
    out = {}
    for sym in df_chunk.columns.levels[0]:
        df = df_chunk[sym].reindex(calendar)
        if df.isna().mean().max() > missing:
            continue
        df = df.ffill().bfill().tail(window + 30)  # 30‐day buffer
        avg_dv = (df['Close'] * df['Volume']).mean()
        if avg_dv < min_vol:
            continue
        out[sym] = df.tail(window)
    return out

def atomic_parquet_write(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        df.to_parquet(tmp.name)
    os.replace(tmp.name, path)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    # universe sources
    p.add_argument('--index-sources', nargs=2, action='append',
                   default=[
                     ['https://en.wikipedia.org/wiki/List_of_S%26P_500_companies','Symbol'],
                     ['https://en.wikipedia.org/wiki/NASDAQ-100','Ticker'],
                     ['https://en.wikipedia.org/wiki/List_of_S%26P_600_companies','Symbol'],
                   ])
    p.add_argument('--sector-etfs', nargs='+',
                   default=['XLF','XLE','XLK','XLB','XLV','XLI','XLU','XLY','XLC','XLP'])
    # filters & sizing
    p.add_argument('--max-tickers', type=int,   default=1500)
    p.add_argument('--min-cap',     type=float, default=1e9)
    p.add_argument('--min-vol',     type=float, default=5e6)
    p.add_argument('--window',      type=int,   default=1250)
    p.add_argument('--missing',     type=float, default=0.05)
    # download settings
    p.add_argument('--chunk-size',  type=int,   default=200)
    p.add_argument('--jobs',        type=int,   default=4)
    p.add_argument('--retries',     type=int,   default=3)
    p.add_argument('--backoff',     type=int,   default=2)
    # date range
    p.add_argument('--start', type=str, default=None)
    p.add_argument('--end',   type=str, default=None)
    args = p.parse_args()

    # paths
    RAW_DIR       = Path('raw')
    METADATA_FILE = RAW_DIR / 'metadata.csv'
    PARQUET_DIR   = RAW_DIR / 'prices'

    # date range
    end = pd.to_datetime(args.end or date.today()).date()
    start = pd.to_datetime(args.start or (end - timedelta(days=args.window+30))).date()
    calendar = pd.bdate_range(start, end)

    # 1) build & filter universe
    df_meta = build_universe(args.index_sources, args.sector_etfs,
                             args.max_tickers, args.min_cap, args.jobs)
    RAW_DIR.mkdir(exist_ok=True)
    df_meta.to_csv(METADATA_FILE, index=False)
    logger.info(f"Saved metadata for {len(df_meta)} symbols → {METADATA_FILE}")

    # 2) download, clean, store prices
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    universe = df_meta['Ticker'].tolist()
    chunks   = [universe[i:i+args.chunk_size] for i in range(0, len(universe), args.chunk_size)]

    for chunk in tqdm(chunks, desc="Downloading price chunks"):
        df_chunk = download_chunk(chunk, start, end, args.retries, args.backoff)
        cleaned  = clean_and_filter(df_chunk, calendar, args.min_vol, args.missing, args.window)
        for sym, df in cleaned.items():
            path = PARQUET_DIR / f"{sym}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                if existing.index.max() >= calendar[-1]:
                    continue
            atomic_parquet_write(df, path)
            logger.info(f"Saved {sym} ({len(df)} rows)")

    logger.info("Data fetch complete.")

if __name__ == '__main__':
    main()
