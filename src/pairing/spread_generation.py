import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ——— Logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ——— Paths ———
SCRIPT_DIR    = Path(__file__).resolve().parent            # .../src/pairing
PREPROC_DIR   = SCRIPT_DIR.parent / 'data' / 'preprocessed'
PAIR_CAND_FILE= SCRIPT_DIR / 'pair_candidates.csv'
PRICES_FILE   = PREPROC_DIR / 'prices.csv'
OUTPUT_PARAMS = SCRIPT_DIR / 'pair_params.csv'
SPREADS_DIR   = SCRIPT_DIR / 'spreads'

# ——— Parameters ———
MIN_OVERLAP   = 100   # minimum days of overlapping prices


def load_inputs():
    """Load price panel and selected pairs."""
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    pairs  = pd.read_csv(PAIR_CAND_FILE)
    log.info(f"Loaded prices ({prices.shape[0]}×{prices.shape[1]}) and {len(pairs)} pairs")
    return prices, pairs


def estimate_hedge_ratio(series_a: pd.Series, series_b: pd.Series):
    """
    Regress A on B: P^A_t = α + β P^B_t + ε_t, return β.
    """
    # align and drop NaNs
    s1, s2 = series_a.align(series_b, join='inner')
    if len(s1) < MIN_OVERLAP:
        return None, None  # insufficient data
    X = sm.add_constant(s2)
    res = sm.OLS(s1, X).fit()
    beta = res.params[s2.name]
    return beta, s1.index


def compute_spread_and_stats(prices: pd.DataFrame, a: str, b: str, beta: float):
    """
    Given prices[a], prices[b], and beta, compute spread series and its mean/std.
    """
    s1, s2 = prices[a], prices[b]
    spread = s1 - beta * s2
    # drop NA after combination
    spread = spread.dropna()
    mu = spread.mean()
    sigma = spread.std()
    zscore = (spread - mu) / sigma
    df = pd.DataFrame({
        'Date': spread.index,
        'Spread': spread.values,
        'ZScore': zscore.values
    })
    return df, mu, sigma


def main():
    prices, pair_df = load_inputs()
    pair_params = []

    SPREADS_DIR.mkdir(exist_ok=True)
    for _, row in pair_df.iterrows():
        a, b, pval = row['TickerA'], row['TickerB'], row['PValue']
        beta, idx = estimate_hedge_ratio(prices[a], prices[b])
        if beta is None:
            log.warning(f"Skipping {a}-{b}: insufficient overlap")
            continue

        spread_df, mu, sigma = compute_spread_and_stats(prices, a, b, beta)
        # save per-pair spread series
        out_file = SPREADS_DIR / f"{a}_{b}_spread.csv"
        spread_df.to_csv(out_file, index=False)
        log.info(f"Saved spread series for {a}-{b} ({len(spread_df)} rows)")

        pair_params.append({
            'TickerA': a,
            'TickerB': b,
            'Beta': beta,
            'Spread_Mean': mu,
            'Spread_STD': sigma,
            'PValue': pval
        })

    # write parameter summary
    params_df = pd.DataFrame(pair_params)
    params_df.to_csv(OUTPUT_PARAMS, index=False)
    log.info(f"Saved pair parameters ({len(params_df)}) to {OUTPUT_PARAMS}")


if __name__ == '__main__':
    main()
