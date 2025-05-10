import os
import sys
import shutil
import pandas as pd
import concurrent.futures

# Constants
MISSING_THRESHOLD = 0.05  # fraction of allowed missing data per ticker

# Paths
BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'preprocessed')

# Wipe preprocessed directory on each run
if os.path.isdir(PROC_DIR):
    shutil.rmtree(PROC_DIR)
os.makedirs(PROC_DIR, exist_ok=True)

# Validate raw directory
if not os.path.isdir(RAW_DIR):
    sys.exit(f"Raw directory not found: {RAW_DIR}")

# Load metadata
meta_path = os.path.join(RAW_DIR, 'metadata.csv')
if not os.path.isfile(meta_path):
    sys.exit(f"metadata.csv not found in {RAW_DIR}")
md = pd.read_csv(meta_path)
tickers = md['Ticker'].dropna().unique().tolist()

# Function to load adjusted-close series for one ticker
def load_adj(ticker):
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    # Prefer 'Adj Close', fallback to 'Close'
    if 'Adj Close' in df.columns:
        series = df['Adj Close'].copy()
    elif 'Close' in df.columns:
        series = df['Close'].copy()
    else:
        return None
    series.name = ticker
    return series

# Parallel load all series
with concurrent.futures.ThreadPoolExecutor() as executor:
    series_list = list(executor.map(load_adj, tickers))

# Filter out missing series
series_list = [s for s in series_list if isinstance(s, pd.Series) and not s.empty]
if not series_list:
    sys.exit("No valid price series found; preprocessing aborted.")

# Concatenate into DataFrame
adj = pd.concat(series_list, axis=1)

# Ensure business-day index coverage
adj = adj.resample('B').ffill().bfill()

# Drop tickers with excessive missing data
min_count = int((1 - MISSING_THRESHOLD) * len(adj))
adj.dropna(axis=1, thresh=min_count, inplace=True)

# Filter metadata to good tickers and save processed metadata
good = adj.columns.tolist()
processed_md = md[md['Ticker'].isin(good)].copy()
processed_meta_path = os.path.join(PROC_DIR, 'metadata.csv')
processed_md.to_csv(processed_meta_path, index=False)
print(f"Processed metadata saved: {len(processed_md)} tickers")

# Compute daily percent returns and drop first row
returns = adj.pct_change().iloc[1:]

# Winsorize at ±3σ and z-score normalize
sigma = returns.std()
returns = returns.clip(-3 * sigma, 3 * sigma, axis=1)
returns = (returns - returns.mean()) / returns.std()

# Save outputs
prices_out = os.path.join(PROC_DIR, 'prices.csv')
returns_out = os.path.join(PROC_DIR, 'returns.csv')
adj.to_csv(prices_out)
returns.to_csv(returns_out)

print(f"Preprocessing complete: {adj.shape[1]} tickers, {returns.shape[0]} days.")
