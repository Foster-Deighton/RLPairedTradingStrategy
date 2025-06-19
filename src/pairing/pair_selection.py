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
MAX_MARKETCAP_DIFF   = 0.4      # Increased from 0.3 to 0.4 (40% max market cap difference)
MIN_PERIODS          = 80       # Reduced from 100 to 80 days
PVALUE_THRESH        = 0.08     # Increased from 0.05 to 0.08 (more lenient cointegration)
ROLL_WINDOW          = 200      # Reduced from 250 to 200 days
ROLL_STEP            = 20       # unchanged
PERSISTENCE_THRESH   = 0.55     # Reduced from 0.65 to 0.55 (only need 55% of windows to pass)
MIN_HALF_LIFE        = 0.05     # Reduced from 0.1 to 0.05 days
MAX_HALF_LIFE        = 25.0     # Increased from 20.0 to 25.0 days
ADF_PVALUE_THRESH    = 0.08     # Increased from 0.05 to 0.08
SPREAD_SPX_CORR_MAX  = 0.25     # Increased from 0.2 to 0.25
CHOW_ALPHA           = 0.08     # Increased from 0.05 to 0.08
EWMA_SPAN            = 200      # Reduced from 250 to 200

# Minimum score required to consider a pair valid (out of 100)
MIN_PAIR_SCORE = 70

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
def safe_coint(x, y):
    """
    Wrapper around coint that handles edge cases and zero residuals.
    Implements a custom cointegration test that avoids the problematic log-likelihood calculation.
    """
    try:
        # 1. Basic data quality checks
        if len(x) < 2 or len(y) < 2:
            return 1.0
        
        # 2. Check for zero or near-zero values
        std_x = x.std()
        std_y = y.std()
        if std_x < 1e-10 or std_y < 1e-10:
            return 1.0
        
        # 3. Check for perfect collinearity
        corr = x.corr(y)
        if abs(corr) > 0.99:
            return 1.0
        
        # 4. Add small noise to break perfect collinearity if present
        if abs(corr) > 0.95:
            x = x * (1 + np.random.normal(0, 1e-6, len(x)))
        
        # 5. Run OLS regression
        model = sm.OLS(x, sm.add_constant(y)).fit()
        
        # 6. Check residuals
        residuals = model.resid
        if residuals.std() < 1e-10:
            return 1.0
            
        # 7. Check for stationarity of residuals using ADF test
        # This is the key part of cointegration testing
        adf_result = adfuller(residuals, maxlag=1, autolag=None)
        p_value = adf_result[1]
        
        # 8. Additional robustness check
        if not np.isfinite(p_value):
            return 1.0
            
        return p_value
        
    except Exception as e:
        tx.debug(f"Error in safe_coint: {str(e)}")
        return 1.0

def rolling_cointegration_persistence(s1, s2):
    """
    Run Engle-Granger on rolling windows.
    Return True if at least PERSISTENCE_THRESH fraction of windows have p <= PVALUE_THRESH.
    """
    n = len(s1)
    if n < ROLL_WINDOW:
        return False

    # Check for perfect collinearity
    corr = s1.corr(s2)
    if abs(corr) > 0.99:
        tx.debug(f"Perfect collinearity detected (corr={corr:.4f})")
        return False

    pvals = []
    valid_windows = 0
    total_windows = 0
    
    for start in range(0, n - ROLL_WINDOW + 1, ROLL_STEP):
        total_windows += 1
        wnd1 = s1.iloc[start:start + ROLL_WINDOW]
        wnd2 = s2.iloc[start:start + ROLL_WINDOW]
        
        # Additional data quality checks with more lenient thresholds
        std1 = wnd1.std()
        std2 = wnd2.std()
        if std1 < 1e-12 or std2 < 1e-12:  # Reduced from 1e-10
            tx.debug(f"Window {start}: Low std dev (s1={std1:.2e}, s2={std2:.2e})")
            pvals.append(1.0)
            continue
            
        # Check for sufficient price movement with more lenient threshold
        range1 = (wnd1.max() - wnd1.min()) / wnd1.mean()
        range2 = (wnd2.max() - wnd2.min()) / wnd2.mean()
        if range1 < 0.005 or range2 < 0.005:  # Reduced from 0.01
            tx.debug(f"Window {start}: Insufficient price movement (range1={range1:.4f}, range2={range2:.4f})")
            pvals.append(1.0)
            continue

        try:
            # Use the safe cointegration test
            p = safe_coint(wnd1, wnd2)
            if np.isnan(p):
                tx.debug(f"Window {start}: NaN p-value")
                pvals.append(1.0)
            else:
                pvals.append(p)
                valid_windows += 1
        except Exception as e:
            tx.debug(f"Window {start}: Error in cointegration test - {str(e)}")
            pvals.append(1.0)

    if not pvals:
        return False

    # Log summary of window analysis
    tx.debug(f"Window analysis: {valid_windows}/{total_windows} valid windows")
    
    # Only consider windows that passed all quality checks
    if valid_windows < total_windows * 0.5:  # Require at least 50% valid windows
        tx.debug(f"Insufficient valid windows: {valid_windows}/{total_windows}")
        return False

    num_windows = len(pvals)
    passed = sum(1 for p in pvals if p <= PVALUE_THRESH)
    persistence = passed / num_windows
    
    tx.debug(f"Persistence: {passed}/{num_windows} windows passed (threshold={PERSISTENCE_THRESH})")
    return persistence >= PERSISTENCE_THRESH

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

    # Check for perfect collinearity
    if abs(df['x'].corr(df['y'])) > 0.99:
        tx.debug("Perfect collinearity detected in Chow test")
        return False

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
        tx.debug("Invalid denominator in Chow test")
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
def calculate_pair_score(results):
    """
    Calculate a score for a pair based on how well it meets various criteria.
    Returns a score from 0-100.
    """
    score = 0
    weights = {
        'cointegration': 25,
        'half_life': 20,
        'adf': 15,
        'regime_stability': 20,
        'spx_correlation': 20
    }
    
    # Cointegration score (0-25)
    if results['cointegration_passed']:
        score += weights['cointegration']
    elif results['cointegration_persistence'] > 0.4:  # Partial credit
        score += weights['cointegration'] * 0.5
    
    # Half-life score (0-20)
    if results['half_life'] is not None:
        hl = results['half_life']
        if MIN_HALF_LIFE <= hl <= MAX_HALF_LIFE:
            score += weights['half_life']
        else:
            # Partial credit if close to range
            if 0.02 <= hl <= 30:
                score += weights['half_life'] * 0.7
    
    # ADF score (0-15)
    if results['adf_pval'] <= ADF_PVALUE_THRESH:
        score += weights['adf']
    elif results['adf_pval'] <= ADF_PVALUE_THRESH * 1.5:  # Partial credit
        score += weights['adf'] * 0.7
    
    # Regime stability score (0-20)
    if results['hi_coint'] and results['lo_coint']:
        score += weights['regime_stability']
    elif results['hi_coint'] or results['lo_coint']:
        score += weights['regime_stability'] * 0.7
    
    # SPX correlation score (0-20)
    if abs(results['corr_spx']) <= SPREAD_SPX_CORR_MAX:
        score += weights['spx_correlation']
    elif abs(results['corr_spx']) <= SPREAD_SPX_CORR_MAX * 1.2:  # Partial credit
        score += weights['spx_correlation'] * 0.7
    
    return score

def test_pair_regime(args):
    try:
        a, b, prices, regime, spx = args
        results = {
            'TickerA': a,
            'TickerB': b,
            'cointegration_passed': False,
            'cointegration_persistence': 0,
            'half_life': None,
            'adf_pval': 1.0,
            'hi_coint': False,
            'lo_coint': False,
            'corr_spx': 0,
            'score': 0
        }
        
        # align price series
        s1_full, s2_full = prices[a].align(prices[b], join='inner')
        if len(s1_full) < MIN_PERIODS:
            return None

        # --- Rolling/subsample cointegration (Steps 2 & 3) ---
        persistence = rolling_cointegration_persistence(s1_full, s2_full)
        results['cointegration_passed'] = persistence
        results['cointegration_persistence'] = persistence

        # --- Dynamic hedge ratio + spread series (Step 5) ---
        beta_series = compute_dynamic_beta(s1_full, s2_full)
        spread = s1_full - beta_series * s2_full
        spread = spread.dropna()
        if spread.empty:
            return None

        # --- Half-life on dynamic spread (Step 6) ---
        hl = compute_half_life(spread)
        results['half_life'] = hl

        # --- Stationarity check on spread (Step 6 continued) ---
        adf_pval = adfuller(spread, maxlag=1, autolag=None)[1]
        results['adf_pval'] = adf_pval

        # --- Regime overlay (Step 7 continued) ---
        vix_col = regime.columns[1]
        vix_series = regime[vix_col].reindex(s1_full.index).ffill().bfill()
        hi_mask = vix_series >= vix_series.quantile(0.75)
        lo_mask = vix_series <= vix_series.quantile(0.25)

        def regime_coint(mask):
            idx = mask[mask].index
            if len(idx) < MIN_PERIODS // 2:
                return False
            try:
                s1_regime = s1_full.loc[idx]
                s2_regime = s2_full.loc[idx]
                p = safe_coint(s1_regime, s2_regime)
                return p <= PVALUE_THRESH
            except Exception as e:
                return False

        results['hi_coint'] = regime_coint(hi_mask)
        results['lo_coint'] = regime_coint(lo_mask)

        # --- SPX-neutrality ---
        spread_ret = spread.pct_change().dropna()
        spx_ret = spx.reindex(spread_ret.index).pct_change().dropna()
        corr_spx = spread_ret.loc[spx_ret.index].corr(spx_ret)
        results['corr_spx'] = corr_spx

        # Calculate final score
        results['score'] = calculate_pair_score(results)
        
        if results['score'] >= MIN_PAIR_SCORE:
            tx.info(f"Found valid pair: {a}-{b} | score={results['score']:.1f} | hl={hl:.2f} | corr_spx={corr_spx:.4f}")
            return {
                'TickerA': a,
                'TickerB': b,
                'HalfLife': hl,
                'CorrSPX': corr_spx,
                'Score': results['score']
            }
        return None

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
            tx.warning("No pairs passed the scoring threshold")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.sort_values('Score', ascending=False, inplace=True)
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

        # Save all pairs found to pair_candidates_full.csv
        full_out = SCRIPT_DIR / 'pair_candidates_full.csv'
        final_df.to_csv(full_out, index=False)
        tx.info(f"Saved all {len(final_df)} pairs to {full_out}")

        # Save only the most significant pairs to pair_candidates.csv
        # Criteria: top 20 by score, or all with score >= 90
        top_df = final_df[final_df['Score'] >= 90]
        if len(top_df) < 20:
            top_df = final_df.head(20)
        top_out = PAIR_OUT
        top_df.to_csv(top_out, index=False)
        tx.info(f"Saved {len(top_df)} most significant pairs to {top_out}")

    except Exception as e:
        tx.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
