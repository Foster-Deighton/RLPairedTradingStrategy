import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression

from stable_baselines3 import PPO

# --- RL‐Driven Strategy Definition ---
class RLPairsTradingStrategy(bt.Strategy):
    params = (
        ('model_path',       'models/fold_0/best_model.zip'),
        ('lookback',         20),
        ('window',           90),    # match training window
        ('target_vol',       0.10),  # 10% annualized vol target
        ('stop_loss_pct',    0.05),  # 5% stop-loss
        ('max_hold_days',    20),    # maximum holding period
        ('trading_cost',     0.0005), # transaction cost per trade
        ('entry_threshold',  0.1),   # Further reduced entry threshold
        ('exit_threshold',   0.05),  # Further reduced exit threshold
        ('max_pair_allocation', 0.1),  # 10% of capital per pair
    )

    def __init__(self):
        super().__init__()
        
        # Initialize observation window for z-score calculation
        self.obs_window = []
        self.price0_window = []
        self.price1_window = []
        
        # Initialize hedge ratio calculation
        self.hedge_ratio_window = []
        self.current_hedge_ratio = 1.0
        
        # Load RL model
        print(f"\nLoading model from: {self.params.model_path}")
        try:
            self.model = PPO.load(self.params.model_path)
            print("Model loaded successfully")
            print(f"Observation space: {self.model.observation_space}")
            print(f"Action space: {self.model.action_space}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize trade tracking
        self.entry_value = None
        self.entry_date = None
        self.order_log = []
        self.equity_curve = []
        
        # Initialize position sizing
        self.initial_capital = self.broker.getvalue()
        self.position_size = 0.0
        self.max_position_value = self.initial_capital * 0.5  # 50% of capital max
        self.max_pair_value = self.initial_capital * self.params.max_pair_allocation
        
        # Debug counters
        self.debug = {
            'obs_skips': 0,
            'nan_skips': 0,
            'action_counts': {'long': 0, 'short': 0, 'exit': 0, 'none': 0}
        }

        # Load regime data
        base = Path(__file__).resolve().parent.parent
        regime_df = pd.read_csv(base / 'data' / 'preprocessed' / 'regime.csv', index_col=0, parse_dates=True)
        spx_df = pd.read_csv(base / 'data' / 'preprocessed' / 'SPX_returns.csv', index_col=0, parse_dates=True)
        
        # Precompute regime features
        regime_values = regime_df[['SPY', '^VIX', '10y2y_Spread', 'Credit_Spread']]
        self.regime_mean = regime_values.mean()
        self.regime_std = regime_values.std() + 1e-8
        self.regime_features = (regime_values - self.regime_mean) / self.regime_std
        
        # SPX returns for macro multiplier
        spx_ret = spx_df['^GSPC']
        self.spx_mean = spx_ret.mean()
        self.spx_std = spx_ret.std() + 1e-8
        self.spx_ret = (spx_ret - self.spx_mean) / self.spx_std

    def calculate_hedge_ratio(self, price0, price1):
        """Calculate dynamic hedge ratio using rolling regression."""
        self.hedge_ratio_window.append((price0, price1))
        if len(self.hedge_ratio_window) > self.params.window:
            self.hedge_ratio_window.pop(0)
        
        if len(self.hedge_ratio_window) < self.params.window:
            return 1.0
            
        # Convert to numpy arrays for calculation
        prices0 = np.array([p[0] for p in self.hedge_ratio_window])
        prices1 = np.array([p[1] for p in self.hedge_ratio_window])
        
        # Calculate hedge ratio using linear regression
        # Add constant term for regression
        X = np.column_stack([np.ones(len(prices1)), prices1])
        y = prices0
        
        try:
            # Use numpy's least squares solver
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            hedge_ratio = beta[1]  # The slope is our hedge ratio
            
            # More aggressive hedge ratio constraints
            if not (0.05 <= hedge_ratio <= 20.0):  # Allow more extreme ratios
                hedge_ratio = 1.0
                
            return float(hedge_ratio)
        except:
            return 1.0

    def calculate_position_size(self, zscore, price0, price1):
        """Calculate position size based on z-score magnitude and volatility targeting."""
        # More aggressive z-score scaling
        zscore_scale = min(abs(zscore), 4.0)  # Increased from 2.0 to 4.0 for more leverage
        
        # Calculate volatility
        if len(self.price0_window) >= self.params.window:
            returns0 = np.diff(np.log(self.price0_window))
            returns1 = np.diff(np.log(self.price1_window))
            vol0 = np.std(returns0) * np.sqrt(252)
            vol1 = np.std(returns1) * np.sqrt(252)
            pair_vol = np.sqrt(vol0**2 + vol1**2)
        else:
            pair_vol = 0.2  # Default to 20% annualized vol if not enough data
        
        # Target volatility (increased from 15% to 30%)
        target_vol = 0.30
        
        # Calculate position size
        if pair_vol > 1e-9:
            # Scale by inverse volatility
            vol_scale = target_vol / pair_vol
            # Combine z-score and volatility scaling
            position_value = self.initial_capital * zscore_scale * vol_scale
            # Cap at maximum position value (increased from 75% to 150%)
            position_value = min(position_value, self.max_pair_value)
            
            # Calculate number of shares
            size0 = int(position_value / (2 * price0))  # Split position between both stocks
            size1 = int(position_value / (2 * price1))
            
            # Ensure minimum position size
            size0 = max(1, size0)
            size1 = max(1, size1)
            
            return size0, size1
        else:
            return 1, 1  # Minimum position size

    def next(self):
        # Get current prices
        price0 = float(self.data0.close[0])
        price1 = float(self.data1.close[0])
        
        # Calculate dynamic hedge ratio
        self.current_hedge_ratio = self.calculate_hedge_ratio(price0, price1)
        
        # Calculate hedged spread
        spread = price0 - (self.current_hedge_ratio * price1)
        
        # Update price windows
        self.price0_window.append(price0)
        self.price1_window.append(price1)
        if len(self.price0_window) > self.params.window:
            self.price0_window.pop(0)
            self.price1_window.pop(0)
        
        # Only calculate z-score if we have enough data
        if len(self.price0_window) < self.params.window:
            self.debug['obs_skips'] += 1
            return
            
        spread_window = np.array(self.price0_window) - (self.current_hedge_ratio * np.array(self.price1_window))
        mean = np.mean(spread_window)
        std = np.std(spread_window)
        if std > 1e-9:
            zscore = (spread - mean) / std
        else:
            zscore = 0.0
        
        # Skip if z-score is NaN
        if np.isnan(zscore):
            self.debug['nan_skips'] += 1
            return
        
        # Update observation window
        self.obs_window.append(zscore)
        if len(self.obs_window) > self.params.window:
            self.obs_window.pop(0)
        
        # Skip if we don't have enough z-score data
        if len(self.obs_window) < self.params.window:
            self.debug['obs_skips'] += 1
            return
        
        # Create observation array - match model's expected shape
        features = []
        
        # Add pair features
        features.extend([zscore, mean, std])
        
        # Calculate half-life of mean reversion
        if len(spread_window) >= 2:
            spread_returns = np.diff(spread_window)
            spread_lag = spread_window[:-1]
            try:
                # Use numpy's least squares solver
                X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
                beta = np.linalg.lstsq(X, spread_returns, rcond=None)[0]
                half_life = -np.log(2) / beta[1] if beta[1] < 0 else 0
                half_life = min(max(half_life, 0), 252)  # Cap between 0 and 252 days
            except:
                half_life = 0
        else:
            half_life = 0
        features.append(half_life)
        
        # Calculate correlation with market
        if len(spread_window) >= 20:
            try:
                current_date = self.datas[0].datetime.date(0)
                spx_returns = self.spx_ret.loc[:current_date].iloc[-20:].values
                spread_returns = np.diff(spread_window[-21:])
                
                # Check for zero standard deviations
                spx_std = np.std(spx_returns)
                spread_std = np.std(spread_returns)
                
                if spx_std > 1e-9 and spread_std > 1e-9:
                    # Calculate correlation manually to avoid numpy warnings
                    spx_norm = (spx_returns - np.mean(spx_returns)) / spx_std
                    spread_norm = (spread_returns - np.mean(spread_returns)) / spread_std
                    correlation = np.mean(spx_norm * spread_norm)
                else:
                    correlation = 0
                
                if np.isnan(correlation):
                    correlation = 0
            except:
                correlation = 0
        else:
            correlation = 0
        features.append(correlation)
        
        # Calculate volatility regime
        if len(spread_window) >= 20:
            try:
                spread_returns = np.diff(spread_window)
                vol = np.std(spread_returns) * np.sqrt(252)
                vol_ma = np.mean([np.std(spread_returns[-i-20:-i]) * np.sqrt(252) 
                                for i in range(10) if len(spread_returns[-i-20:-i]) >= 20])
                regime = 1 if vol > vol_ma else 0
            except:
                regime = 0
        else:
            regime = 0
        features.append(regime)
        
        # Add regime features at current time
        current_date = self.datas[0].datetime.date(0)
        try:
            regime_features = self.regime_features.loc[current_date].values
            features.extend(regime_features)
        except KeyError:
            regime_features = self.regime_features.iloc[-1].values
            features.extend(regime_features)
        
        # Pad or truncate to match model's expected shape
        obs_len = self.model.observation_space.shape[0]
        obs = np.zeros(obs_len, dtype=np.float32)
        features_to_use = features[:obs_len]
        obs[:len(features_to_use)] = features_to_use
        
        # Skip if observation contains NaN values
        if np.isnan(obs).any():
            self.debug['nan_skips'] += 1
            return
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = float(action[0])
        
        # Calculate position sizes
        size0, size1 = self.calculate_position_size(zscore, price0, price1)
        
        # More aggressive action interpretation
        if not self.position:  # No current position
            if action > 0.01:  # Lowered threshold for long
                # Long spread
                self.entry_value = self.broker.getvalue()
                self.entry_date = self.datas[0].datetime.date(0)
                self.buy(data=self.data0, size=size0)
                self.sell(data=self.data1, size=size1)
                self.debug['action_counts']['long'] += 1
            elif action < 0.01:  # Lowered threshold for short
                # Short spread
                self.entry_value = self.broker.getvalue()
                self.entry_date = self.datas[0].datetime.date(0)
                self.sell(data=self.data0, size=size0)
                self.buy(data=self.data1, size=size1)
                self.debug['action_counts']['short'] += 1
            else:
                self.debug['action_counts']['none'] += 1
        else:  # Have position
            # Check if we should exit based on action or z-score
            should_exit = False
            if self.position.size > 0:  # Long position
                if action < -0.01 or abs(zscore) < self.params.exit_threshold:
                    should_exit = True
            elif self.position.size < 0:  # Short position
                if action > 0.01 or abs(zscore) < self.params.exit_threshold:
                    should_exit = True
            
            # Exit if conditions are met
            if should_exit:
                if self.entry_value is not None:  # Only log if we have an entry value
                    self._log_trade('EXIT', zscore)
                self.close()
                self.entry_value = None
                self.entry_date = None
                self.debug['action_counts']['exit'] += 1

        # Record equity curve
        dt = self.datas[0].datetime.date(0)
        equity = self.broker.getvalue()
        self.equity_curve.append({
            'Date': dt, 
            'Equity': equity, 
            'Z': zscore,
            'Action': action,
            'Position': self.position.size if self.position else 0,
            'HedgeRatio': self.current_hedge_ratio
        })

    def _log_trade(self, exit_type, exit_zscore):
        if self.entry_value is None:
            print(f"Warning: _log_trade called without an entry value. Skipping PnL calculation.")
            return
        dt = self.datas[0].datetime.date(0)
        equity = self.broker.getvalue()
        pnl = equity - self.entry_value
        duration = (dt - self.entry_date).days if self.entry_date else 0
        self.order_log.append({
            'EntryDate': self.entry_date.isoformat() if self.entry_date else None,
            'ExitDate':  dt.isoformat(),
            'PnL':        pnl,
            'Duration':   duration,
            'ExitType':   exit_type,
            'ExitZScore': exit_zscore
        })
        self.entry_value = None
        self.entry_date = dt

    def stop(self):
        # Print debug information
        print("\nDebug Information:")
        print(f"Observation skips: {self.debug['obs_skips']}")
        print(f"NaN skips: {self.debug['nan_skips']}")
        print("\nAction Distribution:")
        for action, count in self.debug['action_counts'].items():
            print(f"{action}: {count}")
        
        # Save equity curve & trades
        self.equity_curve_df = pd.DataFrame(self.equity_curve).set_index('Date')
        self.trades_df = pd.DataFrame(self.order_log)

        # Calculate metrics with safety checks
        daily = self.equity_curve_df['Equity'].pct_change().fillna(0)
        
        # Handle Sharpe ratio calculation
        daily_std = daily.std()
        if daily_std > 1e-9:  # Avoid division by zero
            sharpe = daily.mean() / daily_std * np.sqrt(252)
        else:
            sharpe = 0.0
            
        # Handle drawdown calculation
        if not self.equity_curve_df['Equity'].empty:
            cummax = self.equity_curve_df['Equity'].cummax()
            cummax = cummax.replace(0, 1e-9)
            max_dd = (self.equity_curve_df['Equity'] / cummax - 1).min()
            if self.equity_curve_df['Equity'].iloc[0] > 1e-9:
                total = self.equity_curve_df['Equity'].iloc[-1] / self.equity_curve_df['Equity'].iloc[0] - 1
            else:
                total = 0.0
        else:
            max_dd = 0.0
            total = 0.0

        # Calculate pair-specific metrics
        if self.trades_df is not None and not self.trades_df.empty:
            n_wins = (self.trades_df['PnL'] > 0).sum()
            n_trades = len(self.trades_df)
            win_rate = n_wins / n_trades if n_trades > 0 else 0
            avg_win = self.trades_df[self.trades_df['PnL'] > 0]['PnL'].mean() if n_wins > 0 else 0
            avg_loss = self.trades_df[self.trades_df['PnL'] < 0]['PnL'].mean() if (self.trades_df['PnL'] < 0).sum() > 0 else 0
            avg_duration = self.trades_df['Duration'].mean()
            total_profit = self.trades_df[self.trades_df['PnL'] > 0]['PnL'].sum()
            total_loss = self.trades_df[self.trades_df['PnL'] < 0]['PnL'].sum()
            if total_loss == 0:
                if total_profit > 0:
                    profit_factor = float('inf')  # Infinite profit factor if no losses
                else:
                    profit_factor = 0.0
            else:
                profit_factor = abs(total_profit / total_loss)
            if n_wins == 0 or (self.trades_df['PnL'] < 0).sum() == 0:
                print("Warning: No winning or losing trades for this pair. Profit factor may be misleading.")
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            avg_duration = 0
            profit_factor = 0

        self.metrics = pd.Series({
            'Sharpe':       sharpe,
            'MaxDrawdown':  max_dd,
            'TotalReturn':  total,
            'TradeCount':   len(self.trades_df) if self.trades_df is not None else 0,
            'WinRate':      win_rate,
            'AvgWin':       avg_win,
            'AvgLoss':      avg_loss,
            'AvgDuration':  avg_duration,
            'ProfitFactor': profit_factor
        })

        # Print pair-specific metrics
        print("\nPair Performance Metrics:")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Average Duration: {avg_duration:.1f} days")
        print(f"Profit Factor: {profit_factor:.2f}")

# --- Helpers & Runner ---
def load_data(prices_path):
    return pd.read_csv(prices_path, index_col=0, parse_dates=True)

def prepare_feed(df, ticker):
    data = df[[ticker]].rename(columns={ticker:'close'}).copy()
    data['open']   = data['close'].shift(1).bfill()
    data['high']   = data[['open','close']].max(axis=1)
    data['low']    = data[['open','close']].min(axis=1)
    data['volume'] = 1e6
    return bt.feeds.PandasData(dataname=data, name=ticker)

def run_test(args):
    prices_df, pair, lookback, model_path = args
    a, b = pair
    d0 = prepare_feed(prices_df, a)
    d1 = prepare_feed(prices_df, b)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        RLPairsTradingStrategy,
        model_path=model_path,
        lookback=lookback,
        window=lookback,
        target_vol=0.10,
        stop_loss_pct=0.05,
        max_hold_days=20
    )
    cerebro.adddata(d0)
    cerebro.adddata(d1)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.broker.set_slippage_perc(perc=0.0002)
    cerebro.run()

    strat = cerebro.runstrats[0][0]
    return {
        'equity_curve': strat.equity_curve_df,
        'metrics':      strat.metrics,
        'pair':         (a, b)
    }

def download_sp500_data():
    sp500 = yf.download('^GSPC', start='2021-11-29', end='2025-05-25')
    return sp500['Close']

def main():
    base       = Path(__file__).resolve().parent.parent
    prices_p   = base / 'data' / 'preprocessed' / 'prices.csv'
    pairs_p    = base / 'pairing' / 'pair_candidates.csv'
    model_path = base / 'rl' / 'models' / 'fold_0' / 'best_model.zip'

    # Print debug information
    print(f"\nDebug Information:")
    print(f"Base directory: {base}")
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    print(f"Prices path: {prices_p}")
    print(f"Pairs path: {pairs_p}\n")

    prices_df = load_data(prices_p)
    pairs_df  = pd.read_csv(pairs_p)
    tickers   = set(prices_df.columns)
    valid     = pairs_df[
        pairs_df['TickerA'].isin(tickers) &
        pairs_df['TickerB'].isin(tickers)
    ]
    pairs = list(zip(valid['TickerA'], valid['TickerB']))

    #lookbacks = [5, 10, 20, 25, 30, 35, 40]
    lookbacks = [90]
    tasks = [(prices_df, p, lb, str(model_path)) for p in pairs for lb in lookbacks]

    # Download S&P 500 data
    try:
        sp500_data = download_sp500_data()
        if sp500_data.empty:
            print("S&P 500 data is empty. Skipping S&P 500 curve.")
            sp500_data = None
    except Exception as e:
        print(f"Failed to download S&P 500 data: {e}. Skipping S&P 500 curve.")
        sp500_data = None

    # Store metrics for each lookback period
    lookback_metrics = {}
    
    for lb in lookbacks:
        print(f"\nProcessing lookback period L{lb}...")
        lb_tasks = [t for t in tasks if t[2] == lb]
        with Pool(min(len(lb_tasks), os.cpu_count())) as pool:
            results = pool.map(run_test, lb_tasks)

        cumulative_equity = pd.DataFrame()
        cumulative_metrics = []
        for result in results:
            name = f"{result['pair'][0]}-{result['pair'][1]}"
            cumulative_equity[name] = result['equity_curve']['Equity']
            cumulative_metrics.append(result['metrics'])

        # Calculate portfolio metrics
        metrics_df = pd.DataFrame(cumulative_metrics)
        portfolio_metrics = {
            'Sharpe': metrics_df['Sharpe'].mean(),
            'MaxDrawdown': metrics_df['MaxDrawdown'].mean(),
            'TotalReturn': metrics_df['TotalReturn'].mean(),
            'WinRate': metrics_df['WinRate'].mean(),
            'ProfitFactor': metrics_df['ProfitFactor'].mean(),
            'TradeCount': metrics_df['TradeCount'].mean(),
            'AvgDuration': metrics_df['AvgDuration'].mean()
        }
        lookback_metrics[lb] = portfolio_metrics

        lb_dir = Path(f"results/L{lb}")
        lb_dir.mkdir(parents=True, exist_ok=True)
        cumulative_equity.to_csv(lb_dir / 'cumulative_equity.csv')
        metrics_df.to_csv(lb_dir / 'cumulative_metrics.csv')

        # --- Walk-Forward Optimized Portfolio (Top 20% by Sharpe, InvVol Weighted, Rolling) ---
        rebalance_freq = 21  # rebalance every 21 trading days (approx. monthly)
        dates = cumulative_equity.index
        opt_portfolio_equity = pd.Series(index=dates, dtype=float)
        last_weights = None
        last_selected = None
        for i in range(0, len(dates), rebalance_freq):
            window_end = i + rebalance_freq
            rebalance_date = dates[i]
            # Use all data up to (but not including) rebalance_date for stats
            if i == 0:
                # Can't rebalance at the very start, skip
                continue
            window_slice = slice(None, rebalance_date)
            eq_hist = cumulative_equity.loc[window_slice]
            # Compute Sharpe for each pair up to this point
            sharpes = eq_hist.pct_change().mean() / (eq_hist.pct_change().std() + 1e-9) * np.sqrt(252)
            top_pct = 0.2
            n_top = max(1, int(len(sharpes) * top_pct))
            top_pairs = sharpes.sort_values(ascending=False).head(n_top).index
            # Compute inverse vol weights
            vols = eq_hist[top_pairs].pct_change().std() * np.sqrt(252)
            inv_vol = 1 / (vols + 1e-8)
            weights = inv_vol / inv_vol.sum()
            max_alloc = 0.1
            weights = weights.clip(upper=max_alloc)
            weights = weights / weights.sum()
            # Store for use in next period
            last_weights = weights
            last_selected = top_pairs
            # Apply these weights for the next period
            period_slice = slice(dates[i], dates[min(window_end, len(dates)-1)])
            eq_period = cumulative_equity.loc[period_slice, last_selected]
            # If period is shorter than rebalance_freq at end, handle gracefully
            if eq_period.empty:
                continue
            # Compute weighted portfolio for this period
            port_eq = (eq_period * last_weights).sum(axis=1)
            opt_portfolio_equity.loc[eq_period.index] = port_eq
        # Forward fill any missing values (e.g., at the start)
        opt_portfolio_equity = opt_portfolio_equity.ffill().bfill()
        opt_ret = (opt_portfolio_equity / opt_portfolio_equity.iloc[0] - 1) * 100
        # Compute walk-forward optimized portfolio metrics
        opt_daily = opt_portfolio_equity.pct_change().fillna(0)
        opt_sharpe = opt_daily.mean() / (opt_daily.std() + 1e-9) * np.sqrt(252)
        opt_total_return = opt_portfolio_equity.iloc[-1] / opt_portfolio_equity.iloc[0] - 1
        opt_max_dd = (opt_portfolio_equity / opt_portfolio_equity.cummax() - 1).min()
        print("\nWalk-Forward Optimized Portfolio Metrics (Top 20% by Sharpe, Inverse Volatility Weighted, Monthly Rebal):")
        print(f"Sharpe:        {opt_sharpe:.2f}")
        print(f"Total Return:  {opt_total_return:.2%}")
        print(f"Max Drawdown:  {opt_max_dd:.2%}")

        # Save walk-forward optimized portfolio as the main cumulative portfolio result
        # Save equity curve
        opt_portfolio_equity_df = pd.DataFrame({'Equity': opt_portfolio_equity})
        opt_portfolio_equity_df.to_csv(lb_dir / 'cumulative_equity.csv')
        # Save metrics
        opt_metrics = pd.Series({
            'Sharpe': opt_sharpe,
            'MaxDrawdown': opt_max_dd,
            'TotalReturn': opt_total_return,
            'TradeCount': metrics_df['TradeCount'].mean(),  # Still use mean trade count for info
            'WinRate': metrics_df['WinRate'].mean(),
            'ProfitFactor': metrics_df['ProfitFactor'].mean(),
            'AvgDuration': metrics_df['AvgDuration'].mean()
        })
        opt_metrics.to_frame().T.to_csv(lb_dir / 'cumulative_metrics.csv', index=False)
        # Save walk-forward plot as the main cumulative equity plot
        plt.figure(figsize=(10,6))
        # Plot all individual pair returns (faint lines)
        for col in cumulative_equity:
            ret = (cumulative_equity[col] / cumulative_equity[col].iloc[0] - 1) * 100
            plt.plot(cumulative_equity.index, ret, label=col, alpha=0.3)
        # Plot walk-forward optimized portfolio
        plt.plot(opt_portfolio_equity.index, opt_ret, 'k-', linewidth=3.5, label='Walk-Forward\nOptimized Portfolio\n(Top 20%, InvVol Wtd)')
        # Plot S&P 500 if available
        if sp500_data is not None:
            start = opt_portfolio_equity.index[0]
            sp = sp500_data[start:]
            sp_ret = (sp / sp.iloc[0] - 1) * 100
            plt.plot(sp_ret.index, sp_ret, 'k--', label='S&P 500', color='gray', linewidth=3.5)
        plt.title(f'Cumulative Return % for L{lb}')
        plt.xlabel('Date')
        plt.xlim([pd.to_datetime('2021-11-19'), pd.to_datetime('2025-05-29')])
        plt.ylabel('Cumulative Return %')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(lb_dir / 'cumulative_equity_plot.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Optionally, plot equal-weighted portfolio for reference (not as main result)
        plt.figure(figsize=(10,6))
        for col in cumulative_equity:
            ret = (cumulative_equity[col] / cumulative_equity[col].iloc[0] - 1) * 100
            plt.plot(cumulative_equity.index, ret, label=col, alpha=0.3)
        portfolio_ret = cumulative_equity.mean(axis=1)
        portfolio_ret = (portfolio_ret / portfolio_ret.iloc[0] - 1) * 100
        plt.plot(portfolio_ret.index, portfolio_ret, 'k-', linewidth=3.5, label='Equal-Weighted Portfolio')
        if sp500_data is not None:
            start = cumulative_equity.index[0]
            sp = sp500_data[start:]
            sp_ret = (sp / sp.iloc[0] - 1) * 100
            plt.plot(sp_ret.index, sp_ret, '--', label='S&P 500', color='gray', linewidth=3.5)
        plt.title(f'Equal-Weighted Portfolio Cumulative Return % for L{lb}')
        plt.xlabel('Date')
        plt.xlim([pd.to_datetime('2021-11-19'), pd.to_datetime('2025-05-29')])
        plt.ylabel('Cumulative Return %')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(lb_dir / 'equal_weighted_equity_plot.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Update lookback_metrics to use walk-forward optimized portfolio metrics
        lookback_metrics[lb] = opt_metrics.to_dict()

    # Print lookback period comparison
    print("\nLookback Period Comparison:")
    print("=" * 80)
    print(f"{'Lookback':^10} {'Sharpe':^10} {'Return':^10} {'WinRate':^10} {'ProfitFactor':^12} {'Trades':^8} {'Duration':^10}")
    print("-" * 80)
    for lb in sorted(lookback_metrics.keys()):
        metrics = lookback_metrics[lb]
        print(f"{f'L{lb}':^10} {metrics['Sharpe']:^10.2f} {metrics['TotalReturn']:^10.2%} "
              f"{metrics['WinRate']:^10.2%} {metrics['ProfitFactor']:^12.2f} "
              f"{metrics['TradeCount']:^8.0f} {metrics['AvgDuration']:^10.1f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
