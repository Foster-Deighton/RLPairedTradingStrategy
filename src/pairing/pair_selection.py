import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from statsmodels.tsa.stattools import coint

# ——— Logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
tx = logging.getLogger(__name__)

# ——— Paths ———
SCRIPT_DIR   = Path(__file__).resolve().parent              # .../rl_pairs_trading/src/pairing
SRC_DIR      = SCRIPT_DIR.parent                             # .../rl_pairs_trading/src
PREPROC_DIR  = SRC_DIR / 'data' / 'preprocessed'             # cleaned inputs
PAIR_OUT_DIR = SCRIPT_DIR                                    # output directory
PRICES_CSV   = PREPROC_DIR / 'prices.csv'
RETURNS_CSV  = PREPROC_DIR / 'returns.csv'
META_CSV     = PREPROC_DIR / 'metadata.csv'
PAIR_OUT     = PAIR_OUT_DIR / 'pair_candidates.csv'

# ——— Parameters ———
MAX_MARKETCAP_DIFF   = 0.3    # 30% max market cap difference
MIN_PERIODS          = 100    # minimum days for cointegration
CORR_THRESHOLD       = 0.8    # min return-correlation to prefilter
SIGNIFICANCE_LEVEL   = 0.05   # only keep pairs with p-value <= this


def load_preprocessed_data():
    """Load aligned prices, z-scored returns, and metadata."""
    assert PREPROC_DIR.exists(), f"{PREPROC_DIR} not found – run preprocess.py first"
    prices  = pd.read_csv(PRICES_CSV,  index_col=0, parse_dates=True)
    returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
    meta    = pd.read_csv(META_CSV)
    tx.info(f"Loaded {prices.shape[0]} days × {prices.shape[1]} tickers")
    return prices, returns, meta


def filter_pairs(metadata: pd.DataFrame, returns: pd.DataFrame):
    """
    Prefilter ticker pairs by:
      1) same Sector
      2) |MarketCap_A - MarketCap_B| / max(...) <= MAX_MARKETCAP_DIFF
      3) Pearson correlation of returns >= CORR_THRESHOLD
    """
    corr_mat   = returns.corr().values
    tickers    = list(returns.columns)
    idx_map    = {t: i for i, t in enumerate(tickers)}
    candidates = []

    for sector, grp in metadata.groupby('Sector'):
        sect_ts = grp['Ticker'].tolist()
        caps    = grp.set_index('Ticker')['MarketCap'].astype(float)
        avail   = [t for t in sect_ts if t in idx_map]
        n = len(avail)
        if n < 2:
            continue

        # test every pair in this sector
        for i in range(n):
            for j in range(i+1, n):
                a, b = avail[i], avail[j]
                ca, cb = caps[a], caps[b]
                if ca == 0 or cb == 0:
                    continue
                if abs(ca - cb) / max(ca, cb) > MAX_MARKETCAP_DIFF:
                    continue
                if corr_mat[idx_map[a], idx_map[b]] < CORR_THRESHOLD:
                    continue
                candidates.append((a, b))

    tx.info(f"Filtered to {len(candidates)} candidate pairs")
    return candidates


def _test_cointegration(args):
    """Worker: run Engle–Granger test on two price series."""
    a, b, prices = args
    s1, s2 = prices[a].align(prices[b], join='inner')
    if len(s1) < MIN_PERIODS:
        return None
    try:
        _, pval, _ = coint(s1, s2)
        return {'TickerA': a, 'TickerB': b, 'PValue': pval}
    except Exception:
        return None


def compute_cointegration(prices: pd.DataFrame, pairs: list):
    """Parallel cointegration tests, sorted by PValue ascending."""
    if not pairs:
        tx.warning("No candidate pairs to test.")
        return pd.DataFrame(columns=['TickerA','TickerB','PValue'])

    args = [(a, b, prices) for a, b in pairs]
    results = []
    with ProcessPoolExecutor() as exe:
        for res in exe.map(_test_cointegration, args):
            if res:
                results.append(res)

    if not results:
        tx.warning("No cointegration results returned.")
        return pd.DataFrame(columns=['TickerA','TickerB','PValue'])

    df = pd.DataFrame(results)
    df.sort_values('PValue', inplace=True)
    tx.info(f"Computed cointegration for {len(df)} pairs")
    return df


def main():
    prices, returns, metadata = load_preprocessed_data()
    candidate_pairs = filter_pairs(metadata, returns)
    ci_df = compute_cointegration(prices, candidate_pairs)

    # apply significance filter
    sig_df = ci_df[ci_df['PValue'] <= SIGNIFICANCE_LEVEL].reset_index(drop=True)
    tx.info(f"{len(sig_df)} pairs pass p-value ≤ {SIGNIFICANCE_LEVEL}")

    PAIR_OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig_df.to_csv(PAIR_OUT, index=False)
    tx.info(f"Saved {len(sig_df)} significant pairs to {PAIR_OUT}")
    print(sig_df.head(10))


if __name__ == '__main__':
    main()
