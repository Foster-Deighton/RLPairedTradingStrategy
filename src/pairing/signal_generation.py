import logging
from pathlib import Path

import pandas as pd
import numpy as np

# ——— Logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ——— Paths ———
SCRIPT_DIR   = Path(__file__).resolve().parent            # .../rl_pairs_trading/src/pairing
SPREADS_DIR  = SCRIPT_DIR / 'spreads'                     # contains A_B_spread.csv files
SIGNALS_DIR  = SCRIPT_DIR / 'signals'                     # where we’ll write signals

# ——— Parameters ———
ENTRY_THRESHOLD = 2.0   # |z-score| ≥ this triggers entry
EXIT_THRESHOLD  = 0.5   # |z-score| ≤ this triggers exit

def generate_signals():
    """Reads each spread file, computes entry/exit signals, and saves results."""
    if not SPREADS_DIR.exists():
        log.error(f"No spreads folder found at {SPREADS_DIR}; run spread_generation.py first.")
        return

    SIGNALS_DIR.mkdir(exist_ok=True)
    files = list(SPREADS_DIR.glob('*_spread.csv'))
    log.info(f"Found {len(files)} spread files to process.")

    for file in files:
        df = pd.read_csv(file, parse_dates=['Date'])
        # compute boolean flags
        df['LongEntry']  = df['ZScore'] < -ENTRY_THRESHOLD
        df['ShortEntry'] = df['ZScore'] >  ENTRY_THRESHOLD
        df['Exit']       = df['ZScore'].abs() <= EXIT_THRESHOLD
        # optional: a single signal column:  1=long, -1=short, 0=flat/exit
        df['Signal'] = np.where(df['LongEntry'],  1,
                         np.where(df['ShortEntry'], -1,
                         np.where(df['Exit'],         0,
                                                     np.nan)))
        # write out
        out_file = SIGNALS_DIR / file.name.replace('_spread.csv', '_signals.csv')
        df.to_csv(out_file, index=False)
        log.info(f"Wrote signals to {out_file.name}")

    log.info("Signal generation complete.")

if __name__ == '__main__':
    generate_signals()
