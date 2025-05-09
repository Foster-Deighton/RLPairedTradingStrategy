import yfinance as yf
import pandas as pd
import asyncio
import os
import shutil
import logging
from datetime import datetime, timedelta
import time
import concurrent.futures

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SLIDING_WINDOW_LENGTH = 90  # days of history to retain for each ticker
MISSING_THRESHOLD = 0.05    # max allowed fraction of missing data per ticker
CHUNK_SIZE = 200            # maximum tickers per yfinance download call
BUFFER_DAYS = 10            # extra days to ensure full window after gap-fill


def get_top_1000_tickers_by_market_cap():
    """
    Fetch S&P 500 and NASDAQ-100 tickers from Wikipedia,
    then fetch their market caps, industries, and sectors via yfinance,
    and return the top 1000 by market cap.
    Returns:
        tickers (list), industry_map (dict), sector_map (dict), market_caps (dict)
    """
    # Step 1: collect universe from Wikipedia
    try:
        sp500 = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        )[0]['Symbol'].tolist()
        nasdaq = pd.read_html(
            'https://en.wikipedia.org/wiki/NASDAQ-100'
        )[4]['Ticker'].tolist()
        universe = list(set(sp500 + nasdaq))
        logger.info(f"Fetched {len(universe)} tickers from Wikipedia lists")
    except Exception as e:
        logger.error(f"Failed to fetch ticker universe: {e}")
        return [], {}, {}, {}

    # Step 2: fetch metadata in parallel using yfinance
    def fetch_meta(ticker):
        try:
            info = yf.Ticker(ticker).info
            return (
                info.get('marketCap', 0),
                info.get('industry', 'Unknown'),
                info.get('sector', 'Unknown')
            )
        except Exception as e:
            logger.warning(f"Metadata fetch failed for {ticker}: {e}")
            return 0, 'Unknown', 'Unknown'

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(fetch_meta, t): t for t in universe}
        for future in concurrent.futures.as_completed(future_map):
            t = future_map[future]
            try:
                cap, industry, sector = future.result()
            except Exception as e:
                logger.warning(f"Metadata future exception for {t}: {e}")
                cap, industry, sector = 0, 'Unknown', 'Unknown'
            results[t] = (cap, industry, sector)

    market_caps = {t: v[0] for t, v in results.items()}
    industry_map = {t: v[1] for t, v in results.items()}
    sector_map = {t: v[2] for t, v in results.items()}

    sorted_univ = sorted(universe, key=lambda t: market_caps.get(t, 0), reverse=True)
    top = sorted_univ[:1000]
    logger.info(f"Selected top {len(top)} tickers by market cap")
    return top, industry_map, sector_map, market_caps


def download_chunks(tickers, start_date, end_date):
    """
    Download data in chunks via yfinance, clean, gap-fill, and apply sliding window.
    """
    results = {}
    full_idx = pd.date_range(start=start_date, end=end_date, freq='B')
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i+CHUNK_SIZE]
        df_chunk = None
        retries, delay = 3, 1
        for attempt in range(1, retries+1):
            try:
                df_chunk = yf.download(
                    chunk,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    threads=True,
                    auto_adjust=True,
                    actions=False
                )
                break
            except Exception as e:
                logger.warning(
                    f"Chunk {i//CHUNK_SIZE+1} download error (attempt {attempt}): {e}, retrying in {delay}s"
                )
                time.sleep(delay)
                delay *= 2
        if df_chunk is None:
            logger.error(f"Failed to download chunk {i//CHUNK_SIZE+1}")
            continue
        for t in chunk:
            df = df_chunk.get(t)
            if df is None or df.empty:
                logger.warning(f"No data for {t}")
                continue
            df.index = pd.to_datetime(df.index)
            df = df.reindex(full_idx)
            if df.isna().mean().max() > MISSING_THRESHOLD:
                logger.warning(f"{t} >{MISSING_THRESHOLD:.0%} missing; skipping")
                continue
            results[t] = df.ffill().bfill().tail(SLIDING_WINDOW_LENGTH)
    return results


async def fetch_yahoo_data_async(tickers, start_date, end_date):
    """
    Offload download to thread executor.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, download_chunks, tickers, start_date, end_date)


def save_to_csv(data, folder):
    """
    Parallel CSV writes to folder.
    """
    os.makedirs(folder, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = {
            pool.submit(df.to_csv, os.path.join(folder, f"{t}.csv")): t
            for t, df in data.items()
        }
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                future.result()
                logger.info(f"Saved {t}")
            except Exception as e:
                logger.error(f"Error saving {t}: {e}")


async def fetch_and_save(tickers, start_date, end_date, folder):
    data = await fetch_yahoo_data_async(tickers, start_date, end_date)
    save_to_csv(data, folder)
    logger.info("Fetch and save complete.")


if __name__ == '__main__':
    raw_folder = 'raw'
    if os.path.exists(raw_folder):
        shutil.rmtree(raw_folder)
    os.makedirs(raw_folder, exist_ok=True)

    tickers, ind_map, sec_map, mcap = get_top_1000_tickers_by_market_cap()
    if not tickers:
        logger.error("No tickers found; exiting.")
        exit(1)

    # Save metadata
    md = pd.DataFrame({
        'Ticker': tickers,
        'Industry': [ind_map.get(t, 'Unknown') for t in tickers],
        'Sector': [sec_map.get(t, 'Unknown') for t in tickers],
        'MarketCap': [mcap.get(t, 0) for t in tickers],
    })
    md.to_csv(os.path.join(raw_folder, 'metadata.csv'), index=False)
    logger.info(f"Metadata saved to {raw_folder}/metadata.csv")

    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=SLIDING_WINDOW_LENGTH + BUFFER_DAYS)

    asyncio.run(fetch_and_save(tickers, start_date, end_date, raw_folder))
