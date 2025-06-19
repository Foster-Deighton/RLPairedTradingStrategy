import logging
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy.stats import f as f_dist
import yfinance as yf

# ——— Logging ———
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
tx = logging.getLogger(__name__)

# ——— Paths ———
SCRIPT_DIR   = Path(__file__).resolve().parent
SRC_DIR      = SCRIPT_DIR.parent
PREPROC_DIR  = SRC_DIR / 'data' / 'preprocessed'
PAIR_OUT     = SCRIPT_DIR / 'pair_candidates.csv'

PRICES_CSV   = PREPROC_DIR / 'prices.csv'
RETURNS_CSV  = PREPROC_DIR / 'returns.csv'
META_CSV     = PREPROC_DIR / 'metadata.csv'
REGIME_CSV   = PREPROC_DIR / 'regime.csv'
SPX_CSV      = PREPROC_DIR / 'SPX_returns.csv'

# ——— Parameters ———
MAX_MARKETCAP_DIFF   = 0.3      # 30% max market cap difference (Step 1)
MIN_PERIODS          = 100      # minimum days overlap
PVALUE_THRESH        = 0.05     # cointegration p-value cutoff
ROLL_WINDOW          = 250      # window size for rolling cointegration (Step 2)
ROLL_STEP            = 20       # step size for rolling cointegration (Step 2)
PERSISTENCE_THRESH   = 0.8      # require 80% of windows to pass (Step 3)
MIN_HALF_LIFE        = 0.1      # days (Step 6)
MAX_HALF_LIFE        = 20.0     # days (Step 6)
ADF_PVALUE_THRESH    = 0.05     # ADF test cutoff
SPREAD_SPX_CORR_MAX  = 0.2      # max corr(spread, SPX)
CHOW_ALPHA           = 0.05     # significance for Chow structural break (Step 4)
EWMA_SPAN            = 250      # span for EW moving estimates (Step 5)

# ——— Load data ———
def load_preprocessed_data():
    try:
        prices  = pd.read_csv(PRICES_CSV,  index_col=0, parse_dates=True)
        returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
        meta    = pd.read_csv(META_CSV)
        regime  = pd.read_csv(REGIME_CSV, index_col=0, parse_dates=True)

        # Handle SPX data - try both possible column names
        spx_df = pd.read_csv(SPX_CSV, index_col=0, parse_dates=True)
        if '^GSPC' in spx_df.columns:
            spx = spx_df['^GSPC']
        elif 'SPX_Return' in spx_df.columns:
            spx = spx_df['SPX_Return']
        else:
            raise ValueError(f"SPX data must contain either '^GSPC' or 'SPX_Return' column. Found columns: {spx_df.columns.tolist()}")

        # Verify data integrity
        if prices.empty or returns.empty or meta.empty or regime.empty or spx.empty:
            raise ValueError("One or more data files are empty")

        # Check for common date range
        common_dates = prices.index.intersection(regime.index)
        if len(common_dates) < MIN_PERIODS:
            raise ValueError(f"Insufficient common dates between prices and regime data: {len(common_dates)} < {MIN_PERIODS}")

        tx.info(f"Prices {prices.shape}, Returns {returns.shape}, Regime {regime.shape}")
        tx.debug(f"Regime columns: {regime.columns.tolist()}")
        tx.debug(f"Common dates: {len(common_dates)} from {common_dates[0]} to {common_dates[-1]}")

        return prices, returns, meta, regime, spx
    except Exception as e:
        tx.error(f"Error loading data: {str(e)}")
        raise

# ——— Step 1: sector + market-cap prefilter ———
def filter_pairs(metadata):
    try:
        candidates = []
        for sector, grp in metadata.groupby('Sector'):
            ts = grp['Ticker'].tolist()
            caps = grp.set_index('Ticker')['MarketCap'].astype(float)
            for i in range(len(ts)):
                for j in range(i+1, len(ts)):
                    a, b = ts[i], ts[j]
                    ca, cb = caps[a], caps[b]
                    if ca <= 0 or cb <= 0:
                        continue
                    if abs(ca - cb) / max(ca, cb) > MAX_MARKETCAP_DIFF:
                        continue
                    candidates.append((a, b))
        tx.info(f"Prefiltered to {len(candidates)} intra‐sector pairs")
        return candidates
    except Exception as e:
        tx.error(f"Error in filter_pairs: {str(e)}")
        raise

# ——— Step 2 (and 3): rolling/subsample cointegration + persistence check ———
def rolling_cointegration_persistence(s1, s2):
    """
    Run Engle-Granger on rolling windows.
    Return True if at least PERSISTENCE_THRESH fraction of windows have p <= PVALUE_THRESH.
    """
    n = len(s1)
    if n < ROLL_WINDOW:
        return False

    pvals = []
    for start in range(0, n - ROLL_WINDOW + 1, ROLL_STEP):
        wnd1 = s1.iloc[start:start + ROLL_WINDOW]
        wnd2 = s2.iloc[start:start + ROLL_WINDOW]
        try:
            p = coint(wnd1, wnd2)[1]
            pvals.append(p)
        except Exception:
            pvals.append(1.0)  # treat failures as non-cointegrated
    if not pvals:
        return False

    num_windows = len(pvals)
    passed = sum(1 for p in pvals if p <= PVALUE_THRESH)
    return (passed / num_windows) >= PERSISTENCE_THRESH

# ——— Step 4: Chow structural break test ———
def chow_test(s1, s2):
    """
    Perform a Chow test at midpoint to check if the regression s1 ~ s2 changes slope or intercept.
    Returns True if NO significant break (i.e., we keep pair); False if a break is detected.
    """
    df = pd.DataFrame({'y': s1.values, 'x': s2.values})
    df = df.dropna()
    n = len(df)
    if n < 2 * ROLL_WINDOW:
        # not enough data for a reliable split at midpoint
        return True

    mid = n // 2
    df_full = sm.add_constant(df['x'])
    model_full = sm.OLS(df['y'], df_full).fit()
    rss_full = sum(model_full.resid ** 2)

    # first half
    df1 = sm.add_constant(df['x'][:mid])
    model1 = sm.OLS(df['y'][:mid], df1).fit()
    rss1 = sum(model1.resid ** 2)

    # second half
    df2 = sm.add_constant(df['x'][mid:])
    model2 = sm.OLS(df['y'][mid:], df2).fit()
    rss2 = sum(model2.resid ** 2)

    # number of parameters = 2 (intercept and slope)
    k = 2
    # compute F-statistic
    numerator = (rss_full - (rss1 + rss2)) / k
    denominator = (rss1 + rss2) / (n - 2 * k)
    if denominator <= 0:
        return True  # cannot compute; treat as no break
    f_stat = numerator / denominator
    # degrees of freedom
    df1_f = k
    df2_f = n - 2 * k
    p_value = 1 - f_dist.cdf(f_stat, df1_f, df2_f)
    # if p < CHOW_ALPHA, there's a structural break
    return not (p_value < CHOW_ALPHA)

# ——— Step 5: dynamic hedge ratio via EW moving estimates ———
def compute_dynamic_beta(s1, s2):
    """
    Compute an exponentially-weighted moving beta = cov(s1,s2) / var(s2).
    Returns a pandas Series of beta aligned with s1/s2 index.
    """
    s1_mean = s1.ewm(span=EWMA_SPAN).mean()
    s2_mean = s2.ewm(span=EWMA_SPAN).mean()
    s1s2_mean = (s1 * s2).ewm(span=EWMA_SPAN).mean()
    s2s2_mean = (s2 * s2).ewm(span=EWMA_SPAN).mean()

    cov = s1s2_mean - s1_mean * s2_mean
    var_s2 = s2s2_mean - s2_mean * s2_mean
    beta = cov / var_s2
    return beta

# ——— Step 6: half-life calculation on dynamic spread ———
def compute_half_life(spread):
    """
    Given a spread series, compute phi and half-life:
       phi = lagged regression coefficient of Δspread on lagged spread.
       half-life = -ln(2) / ln(|phi|)
    Returns half-life (float) or None if invalid.
    """
    spread = spread.dropna()
    lagged = spread.shift(1).dropna()
    delta = (spread - lagged).dropna()
    lagged, delta = lagged.align(delta, join='inner')
    if len(lagged) == 0:
        return None

    phi = np.dot(lagged, delta) / np.dot(lagged, lagged)
    if not (0 < abs(phi) < 1):
        return None
    return -np.log(2) / np.log(abs(phi))

# ——— Step 7: regime‐stable cointegration test + full‐pipeline for each pair ———
def test_pair_regime(args):
    try:
        a, b, prices, regime, spx = args
        # align price series
        s1_full, s2_full = prices[a].align(prices[b], join='inner')
        if len(s1_full) < MIN_PERIODS:
            tx.debug(f"Pair {a}-{b}: Insufficient data points (n={len(s1_full)})")
            return None

        # --- Rolling/subsample cointegration (Steps 2 & 3) ---
        if not rolling_cointegration_persistence(s1_full, s2_full):
            tx.debug(f"Pair {a}-{b}: Failed persistent cointegration check")
            return None

        # --- Structural-break test on static regression (Step 4) ---
        if not chow_test(s1_full, s2_full):
            tx.debug(f"Pair {a}-{b}: Detected structural break in hedge ratio")
            return None

        # --- Dynamic hedge ratio + spread series (Step 5) ---
        beta_series = compute_dynamic_beta(s1_full, s2_full)
        spread = s1_full - beta_series * s2_full
        spread = spread.dropna()
        if spread.empty:
            tx.debug(f"Pair {a}-{b}: Spread empty after dynamic beta")
            return None

        # --- Half-life on dynamic spread (Step 6) ---
        hl = compute_half_life(spread)
        if hl is None or not (MIN_HALF_LIFE <= hl <= MAX_HALF_LIFE):
            tx.debug(f"Pair {a}-{b}: Half-life {hl} not in [{MIN_HALF_LIFE}, {MAX_HALF_LIFE}]")
            return None

        # --- Stationarity check on spread (Step 6 continued) ---
        adf_pval = adfuller(spread, maxlag=1, autolag=None)[1]
        if adf_pval > ADF_PVALUE_THRESH:
            tx.debug(f"Pair {a}-{b}: Spread failed ADF (p={adf_pval:.4f})")
            return None

        # --- Regime overlay (Step 7 continued) ---
        # Use VIX column as regime proxy
        vix_col = regime.columns[1]
        vix_series = regime[vix_col].reindex(s1_full.index).ffill().bfill()
        hi_mask = vix_series >= vix_series.quantile(0.75)
        lo_mask = vix_series <= vix_series.quantile(0.25)

        def regime_coint(mask):
            idx = mask[mask].index
            if len(idx) < MIN_PERIODS // 2:
                return False
            try:
                p = coint(s1_full.loc[idx], s2_full.loc[idx])[1]
                return p <= PVALUE_THRESH
            except Exception:
                return False

        if not (regime_coint(hi_mask) and regime_coint(lo_mask)):
            tx.debug(f"Pair {a}-{b}: Failed cointegration in at least one regime")
            return None

        # --- Step 7 added: SPX-neutrality (same as original) ---
        spread_ret = spread.pct_change().dropna()
        spx_ret = spx.reindex(spread_ret.index).pct_change().dropna()
        corr_spx = spread_ret.loc[spx_ret.index].corr(spx_ret)
        if abs(corr_spx) > SPREAD_SPX_CORR_MAX:
            tx.debug(f"Pair {a}-{b}: High SPX correlation ({corr_spx:.4f})")
            return None

        tx.info(f"Found valid pair: {a}-{b} | hl={hl:.2f} | corr_spx={corr_spx:.4f}")
        return {
            'TickerA': a,
            'TickerB': b,
            'HalfLife': hl,
            'CorrSPX': corr_spx
        }

    except Exception as e:
        tx.error(f"Error in test_pair_regime for {a}-{b}: {str(e)}")
        return None

def compute_pairs(prices, candidates, regime, spx):
    try:
        args = [(a, b, prices, regime, spx) for a, b in candidates]
        results = []
        with ProcessPoolExecutor() as exe:
            for res in exe.map(test_pair_regime, args):
                if res:
                    results.append(res)

        if not results:
            tx.warning("No pairs passed all filtering steps")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.sort_values('HalfLife', inplace=True)
        tx.info(f"Found {len(df)} final pairs")
        return df
    except Exception as e:
        tx.error(f"Error in compute_pairs: {str(e)}")
        raise

def main():
    try:
        prices, returns, meta, regime, spx = load_preprocessed_data()
        cands = filter_pairs(meta)
        final_df = compute_pairs(prices, cands, regime, spx)

        if final_df.empty:
            tx.error("No pairs passed the full selection pipeline.")
            sys.exit(1)

        final_df.to_csv(PAIR_OUT, index=False)
        tx.info(f"Saved {len(final_df)} pairs to {PAIR_OUT}")

    except Exception as e:
        tx.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
