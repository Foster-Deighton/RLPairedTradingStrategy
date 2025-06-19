import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# ——— Logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ——— Paths ———
SCRIPT_DIR     = Path(__file__).resolve().parent
PREPROC_DIR    = SCRIPT_DIR.parent / 'data' / 'preprocessed'
PAIR_FILE      = SCRIPT_DIR / 'pair_candidates.csv'
PRICES_FILE    = PREPROC_DIR / 'prices.csv'
OUTPUT_PARAMS  = SCRIPT_DIR / 'pair_params.csv'
SPREADS_DIR    = SCRIPT_DIR / 'spreads'

# ——— Parameters ———
MIN_OVERLAP     = 100    # minimum days needed
ROLLING_WINDOW  = 60     # window for dynamic β and z-score
ADF_MAX_LAG     = 1      # for ADF test
ADF_CRIT_LEVEL  = 0.05
                        
def load_data():
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    pairs  = pd.read_csv(PAIR_FILE)
    log.info(f"Loaded prices {prices.shape} and {len(pairs)} pair candidates")
    return prices, pairs

def static_hedge_ratio(s1, s2):
    """OLS β = cov(s1,s2)/var(s2)."""
    df = pd.concat([s1, s2], axis=1).dropna()
    if len(df) < MIN_OVERLAP:
        return None
    x = df.iloc[:,1].values
    y = df.iloc[:,0].values
    return float(np.dot(y, x) / np.dot(x, x))

def dynamic_hedge_ratio(s1, s2):
    """Rolling β via Cov/Var on a fixed window."""
    df = pd.concat([s1, s2], axis=1).dropna()
    cov = df.iloc[:,1].rolling(ROLLING_WINDOW).cov(df.iloc[:,0])
    var = df.iloc[:,1].rolling(ROLLING_WINDOW).var()
    return (cov/var).reindex(s1.index).fillna(method='bfill').fillna(method='ffill')

def compute_half_life(spread):
    """Half-life from an AR(1) fit: Δ=φ·lag + ε."""
    sr = spread.dropna()
    if len(sr) < MIN_OVERLAP:
        return None
    lagged = sr.shift(1).dropna()
    delta  = (sr - lagged).dropna()
    if len(delta) != len(lagged):
        lagged = lagged.loc[delta.index]
    phi = np.dot(lagged.values, delta.values) / np.dot(lagged.values, lagged.values)
    if phi <= 0 or phi >= 1:
        return None
    return float(-np.log(2) / np.log(abs(phi)))

def process_pair(row, prices):
    a, b = row['TickerA'], row['TickerB']
    
    # Check if both tickers exist in prices
    if a not in prices.columns:
        log.warning(f"Ticker {a} not found in price data")
        return None
    if b not in prices.columns:
        log.warning(f"Ticker {b} not found in price data")
        return None
        
    s1, s2 = prices[a], prices[b]
    
    # Check if we have enough data for both stocks
    if s1.isna().all() or s2.isna().all():
        log.warning(f"{a}-{b}: insufficient price data")
        return None
        
    # 1) Static β & static spread
    beta_s = static_hedge_ratio(s1, s2)
    if beta_s is None:
        log.warning(f"{a}-{b}: insufficient data for hedge ratio calculation")
        return None
    spread_s = (s1 - beta_s * s2).dropna()
    # static stats
    mu_s    = spread_s.mean()
    sigma_s = spread_s.std()
    kurt    = spread_s.kurtosis()
    skew    = spread_s.skew()
    hl_s    = compute_half_life(spread_s)
    # ADF test
    try:
        adf_p = adfuller(spread_s, maxlag=ADF_MAX_LAG, autolag=None)[1]
    except Exception as e:
        log.warning(f"{a}-{b}: ADF failed: {e}")
        return None

    # 2) Dynamic β & spread, then rolling z-score
    beta_d = dynamic_hedge_ratio(s1, s2)
    spread_d = (s1 - beta_d * s2).dropna()
    roll_mu  = spread_d.rolling(ROLLING_WINDOW).mean()
    roll_sd  = spread_d.rolling(ROLLING_WINDOW).std()
    z_dyn    = ((spread_d - roll_mu) / roll_sd).dropna()

    # Save per-pair spread CSV
    df_out = pd.DataFrame({
        'Date':      spread_d.index,
        'PriceA':    s1.reindex(spread_d.index),
        'PriceB':    s2.reindex(spread_d.index),
        'BetaDyn':   beta_d.reindex(spread_d.index),
        'SpreadDyn': spread_d,
        'ZScoreDyn': z_dyn
    })
    out_path = SPREADS_DIR / f"{a}_{b}_spread.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"{a}-{b}: wrote {len(df_out)} rows to {out_path.name}")

    # Return summary parameters
    return {
        'TickerA':      a,
        'TickerB':      b,
        'BetaStatic':   beta_s,
        'HalfLife':     hl_s,
        'ADF_pvalue':   adf_p,
        'SigmaStatic':  sigma_s,
        'Kurtosis':     kurt,
        'Skewness':     skew
    }

if __name__ == '__main__':
    prices, pair_df = load_data()
    
    # Log available tickers for debugging
    log.info(f"Available tickers in price data: {len(prices.columns)}")
    log.info(f"Number of pair candidates: {len(pair_df)}")
    
    # Check for missing tickers
    all_tickers = set(pair_df['TickerA'].unique()) | set(pair_df['TickerB'].unique())
    missing_tickers = all_tickers - set(prices.columns)
    if missing_tickers:
        log.warning(f"Missing tickers in price data: {missing_tickers}")
    
    SPREADS_DIR.mkdir(exist_ok=True, parents=True)
    params = []
    with ProcessPoolExecutor() as exe:
        futures = {exe.submit(process_pair, row, prices): row for _, row in pair_df.iterrows()}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    params.append(res)
            except Exception as e:
                log.error(f"Error processing pair: {e}")

    if params:
        pd.DataFrame(params).to_csv(OUTPUT_PARAMS, index=False)
        log.info(f"Saved parameters for {len(params)} pairs to {OUTPUT_PARAMS}")
    else:
        log.error("No valid pairs were processed. Check the warnings above for details.")
