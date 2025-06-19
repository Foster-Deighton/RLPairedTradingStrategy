import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yfinance as yf

# ——— Logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ——— Paths ———
SCRIPT_DIR   = Path(__file__).resolve().parent
SPREADS_DIR  = SCRIPT_DIR / 'spreads'     # contains A_B_spread.csv
SIGNALS_DIR  = SCRIPT_DIR / 'signals'     # output folder

# ——— Parameters ———
RISK_ON_THRESH     = 0.0    # SPX MA50 - MA200 > this => risk_on
ENTRY_Z_ON         = 1.0    # entry threshold in risk-on
EXIT_Z_ON          = 0.2    # exit threshold in risk-on
ENTRY_Z_OFF        = 1.5    # entry threshold in risk-off
EXIT_Z_OFF         = 0.5    # exit threshold in risk-off
PORTFOLIO_VOL_TGT  = 0.01   # target vol per trade (1%)
STOP_LOSS_PCT      = 0.02   # 2% stop-loss
MAX_HOLD_DAYS      = 5      # time-stop

# ——— Helpers ———
def download_spx(start_date, end_date):
    """Download S&P 500 data for the given date range."""
    try:
        # Download the data
        df = yf.download('^GSPC', start=start_date, end=end_date)
        
        # Check if we got any data
        if df.empty:
            log.error("No SPX data downloaded")
            return None
            
        # Check if we have the required column
        if 'Adj Close' not in df.columns:
            log.warning("Adj Close not available, using Close instead")
            if 'Close' not in df.columns:
                log.error("Neither Adj Close nor Close available in SPX data")
                return None
            price_col = 'Close'
        else:
            price_col = 'Adj Close'
            
        # Extract the price column and create a new DataFrame
        df = pd.DataFrame(df[price_col])
        df.columns = ['SPX']
        
        # Normalize the index to remove time component
        df.index = df.index.normalize()
        
        log.info(f"Successfully downloaded SPX data from {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        log.error(f"Failed to download SPX data: {str(e)}")
        return None

def classify_regime(spx_series):
    ma50  = spx_series.rolling(50).mean()
    ma200 = spx_series.rolling(200).mean()
    reg   = (ma50 - ma200) > RISK_ON_THRESH
    return reg.astype(int)  # 1 = risk_on, 0 = risk_off

def process_pair(file_path):
    """Process a single pair's spread data."""
    try:
        # Convert file_path to string if it's a Path object
        file_path_str = str(file_path)
        
        # Load spread data
        df = pd.read_csv(file_path_str, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        
        # Check for required columns
        required_cols = ['SpreadDyn', 'PriceA', 'PriceB', 'BetaDyn']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log.error(f"Missing required columns in {file_path_str}: {missing_cols}")
            return None
        
        # Download SPX data for the same period
        spx = download_spx(df.index.min(), df.index.max())
        if spx is None:
            log.error(f"Failed to process {file_path_str} due to SPX download error")
            return None
            
        # Ensure dates align
        common_dates = df.index.intersection(spx.index)
        if len(common_dates) < 20:  # Minimum required for calculations
            log.error(f"Insufficient overlapping dates between spread and SPX data for {file_path_str}")
            return None
            
        df = df.loc[common_dates]
        spx = spx.loc[common_dates]
            
        # Calculate SPX correlation
        spx_corr = df['SpreadDyn'].corr(spx['SPX'])
        
        # Calculate rolling statistics
        window = 20  # 20-day rolling window
        df['RollingMean'] = df['SpreadDyn'].rolling(window=window).mean()
        df['RollingStd'] = df['SpreadDyn'].rolling(window=window).std()
        df['ZScore'] = (df['SpreadDyn'] - df['RollingMean']) / df['RollingStd']
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['ZScore'] < -2.0, 'Signal'] = 1  # Long signal
        df.loc[df['ZScore'] > 2.0, 'Signal'] = -1  # Short signal
        
        # Save processed data
        output_path = file_path_str.replace('_spread.csv', '_signals.csv')
        df.to_csv(output_path)
        log.info(f"Processed {file_path_str} -> {output_path}")
        
        return {
            'file': file_path_str,
            'spx_correlation': spx_corr,
            'signal_count': len(df[df['Signal'] != 0])
        }
        
    except Exception as e:
        log.error(f"Error processing {file_path_str}: {str(e)}")
        return None

def main():
    # Create signals directory if it doesn't exist
    SIGNALS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Get all spread files
    spread_files = list(Path(SPREADS_DIR).glob('*_spread.csv'))
    if not spread_files:
        log.error("No spread files found!")
        return
        
    log.info(f"Found {len(spread_files)} spread files to process")
    
    # Process files in parallel
    results = []
    with ProcessPoolExecutor() as exe:
        for result in exe.map(process_pair, spread_files):
            if result:
                results.append(result)
    
    # Save summary
    if results:
        summary_df = pd.DataFrame(results)
        output_summary = SIGNALS_DIR / 'signal_summary.csv'
        summary_df.to_csv(output_summary, index=False)
        log.info(f"Processed {len(results)} pairs, summary saved to {output_summary}")
    else:
        log.error("No pairs were successfully processed!")

if __name__ == '__main__':
    main()

