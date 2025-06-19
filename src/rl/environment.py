import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import statsmodels.api as sm

class PairsEnv(gym.Env):
    """
    Gym environment for pairs trading with regime signals and risk-adjusted rewards.

    Observation:
      - For each pair: 
        - [current z-score, rolling mean of spread, rolling std of spread]
        - half-life of mean reversion
        - rolling correlation with market (20-day window)
        - correlation breakdown indicator
        - volatility regime indicator (using GARCH-like approach)
        - regime persistence score
      - Current macro/regime features: SPY 200d momentum, VIX, 10y–2y spread, credit spread

    Action:
      - Box: continuous position in [-1, +1] for each pair (negative = short, positive = long)

    Reward:
      - Scaled PnL based on z-score change and position size
      - Minus drawdown penalty (lambda_dd × (drawdown / window_length))
      - Minus volatility penalty (gamma_vol × |scaled PnL|)
      - Macro‐multiplier: boosts reward for profitable trades in bear, dampens in bull
      - Small time‐step penalty if entirely flat to discourage doing nothing
      - **NEW:** Market-neutrality penalty (lambda_beta × portfolio_beta^2)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_csv: str,
        pairs_csv: str,
        regime_csv: str,
        spx_csv: str,
        window_length: int = 270,
        trading_cost: float = 0.0005,
        beta_spy: float = 1.0,
    ):
        super().__init__()

        # 1) Load all data
        prices_df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
        pairs_df = pd.read_csv(pairs_csv)
        regime_df = pd.read_csv(regime_csv, index_col=0, parse_dates=True)
        spx_df = pd.read_csv(spx_csv, index_col=0, parse_dates=True)

        # Handle missing values in price data using newer methods
        prices_df = prices_df.ffill().bfill()

        # 2) Prepare raw spread arrays and additional features
        self.pairs = list(zip(pairs_df.TickerA, pairs_df.TickerB))
        self.window = window_length
        self.spreads = {}  # Price-based spreads for z-score calculation
        self.return_spreads = {}  # Return-based spreads for actual trading
        self.half_lives = {}
        self.correlations = {}
        self.vol_regimes = {}
        self.regime_persistence = {}
        self.corr_breakdown = {}
        
        # Parameters for feature calculation
        self.corr_window = 20
        self.vol_window = 60
        self.regime_threshold = 0.2
        
        # Store the first valid index for return spreads
        self.return_start_idx = 1  # Because of np.diff
        
        for a, b in self.pairs:
            # Calculate price-based spread for z-scores
            s1 = prices_df[a].values
            s2 = prices_df[b].values
            price_spread = s1 - s2
            self.spreads[(a, b)] = price_spread.astype(np.float32)
            
            # Calculate return-based spread for actual trading
            ret_a = np.diff(s1) / s1[:-1]
            ret_b = np.diff(s2) / s2[:-1]
            ret_a = np.nan_to_num(ret_a, nan=0.0)
            ret_b = np.nan_to_num(ret_b, nan=0.0)
            return_spread = ret_a - ret_b
            self.return_spreads[(a, b)] = return_spread.astype(np.float32)
            
            # Calculate half-life using price-based spread
            try:
                spread_lag = price_spread[:-1]
                spread_ret = np.diff(price_spread)
                model = sm.OLS(spread_ret, sm.add_constant(spread_lag)).fit()
                if len(model.params) >= 2 and model.params[1] < 0:
                    half_life = -np.log(2) / model.params[1]
                else:
                    half_life = 0
            except Exception as e:
                print(f"Warning: Half-life calculation failed for pair {pair}: {e}")
                half_life = 0
            self.half_lives[(a, b)] = half_life
            
            # Calculate rolling correlation with market using return spreads
            spx_returns = spx_df['^GSPC'].pct_change().values[1:]
            spx_returns = np.nan_to_num(spx_returns, nan=0.0)
            min_len = min(len(return_spread), len(spx_returns))
            return_spread = return_spread[:min_len]
            spx_returns = spx_returns[:min_len]
            
            # Initialize rolling correlation array
            rolling_corr = np.zeros(len(return_spread))
            for i in range(self.corr_window, len(return_spread)):
                # Get the window data
                spread_window = return_spread[i-self.corr_window:i]
                spx_window = spx_returns[i-self.corr_window:i]
                
                # Calculate means and standard deviations
                spread_mean = np.mean(spread_window)
                spx_mean = np.mean(spx_window)
                spread_std = np.std(spread_window)
                spx_std = np.std(spx_window)
                
                # Only calculate correlation if both standard deviations are non-zero
                if spread_std > 1e-8 and spx_std > 1e-8:
                    # Calculate correlation manually to avoid numpy warnings
                    spread_centered = spread_window - spread_mean
                    spx_centered = spx_window - spx_mean
                    covariance = np.mean(spread_centered * spx_centered)
                    correlation = covariance / (spread_std * spx_std)
                    rolling_corr[i] = correlation
                else:
                    rolling_corr[i] = 0.0  # Default to 0 if standard deviation is too small
            
            self.correlations[(a, b)] = rolling_corr
            
            # Calculate correlation breakdown indicator
            corr_breakdown = np.zeros(len(rolling_corr))
            corr_breakdown[np.abs(rolling_corr) > 0.5] = 1
            self.corr_breakdown[(a, b)] = corr_breakdown
            
            # Calculate volatility regime using return spreads
            vol = np.zeros(len(return_spread))
            regime = np.zeros(len(return_spread))
            persistence = np.zeros(len(return_spread))
            
            for i in range(self.vol_window, len(return_spread)):
                vol[i] = np.std(return_spread[i-self.vol_window:i]) * np.sqrt(252)
                regime[i] = 1 if vol[i] > self.regime_threshold else 0
                if i > 0:
                    persistence[i] = 0.95 * persistence[i-1] + 0.05 * regime[i]
                else:
                    persistence[i] = regime[i]
            
            self.vol_regimes[(a, b)] = regime
            self.regime_persistence[(a, b)] = persistence

        # 3) Dates index
        self.dates = prices_df.index
        self.n_steps = len(self.dates)

        # 4) Precompute regime features aligned to dates and normalize
        regime_df = regime_df.reindex(self.dates).ffill().bfill()
        regime_values = regime_df[['SPY', '^VIX', '10y2y_Spread', 'Credit_Spread']].values.astype(np.float32)
        self.regime_mean = regime_values.mean(axis=0)
        self.regime_std = regime_values.std(axis=0) + 1e-8
        self.regime_features = (regime_values - self.regime_mean) / self.regime_std

        # 5) SPX returns for macro multiplier
        spx_ret = spx_df['^GSPC'].reindex(self.dates).ffill().bfill().values.astype(np.float32)
        self.spx_mean = spx_ret.mean()
        self.spx_std = spx_ret.std() + 1e-8
        self.spx_ret = (spx_ret - self.spx_mean) / self.spx_std

        # 6) Risk penalty weights
        self.cost = trading_cost
        self.lambda_dd = 0.05
        self.gamma_vol = 0.01
        self.beta_spy = beta_spy
        self.lambda_beta = 0.02

        # 7) Tracking PnL & drawdown
        self.cum_pnl = 0.0
        self.max_cum_pnl = 0.0

        # 8) Observation & action space definitions
        n_pairs = len(self.pairs)
        n_regime = self.regime_features.shape[1]
        obs_dim = n_pairs * 8 + n_regime

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_pairs,),
            dtype=np.float32
        )

        # 9) Initialize state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Start at a random time index allowing for window + one step ahead
        self.t = np.random.randint(self.window, self.n_steps - 1)
        self.prev_pos = np.zeros(len(self.pairs), dtype=np.float32)
        self.cum_pnl = 0.0
        self.max_cum_pnl = 0.0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Build observation with proper normalization and NaN handling
        """
        features = []
        for pair in self.pairs:
            spread = self.spreads[pair]
            seg = spread[self.t - self.window : self.t]
            
            # Handle NaN values in spread
            seg = np.nan_to_num(seg, nan=0.0)
            current_spread = np.nan_to_num(spread[self.t], nan=0.0)
            
            # Calculate z-score with NaN handling
            mean_seg = np.nanmean(seg)
            std_seg = np.nanstd(seg) + 1e-8
            z_t = (current_spread - mean_seg) / std_seg
            
            # Get current correlation and regime indicators with NaN handling
            corr_idx = min(self.t - 1, len(self.correlations[pair]) - 1)
            corr = np.nan_to_num(self.correlations[pair][corr_idx], nan=0.0)
            corr_break = np.nan_to_num(self.corr_breakdown[pair][corr_idx], nan=0.0)
            vol_regime = np.nan_to_num(self.vol_regimes[pair][corr_idx], nan=0.0)
            regime_persist = np.nan_to_num(self.regime_persistence[pair][corr_idx], nan=0.0)
            
            # Normalize features to reasonable ranges
            z_t = np.clip(z_t, -10.0, 10.0)  # Clip extreme z-scores
            mean_seg = np.clip(mean_seg, -1e6, 1e6)  # Clip extreme means
            std_seg = np.clip(std_seg, 1e-8, 1e6)  # Ensure positive std
            half_life = np.clip(self.half_lives[pair], 0, 252)  # Clip to trading days
            corr = np.clip(corr, -1.0, 1.0)  # Ensure correlation is in [-1, 1]
            
            # Add pair features
            features.extend([
                z_t, 
                mean_seg, 
                std_seg,
                half_life,
                corr,
                corr_break,
                vol_regime,
                regime_persist
            ])

        # Append normalized regime features at time t with NaN handling
        reg_feat = np.nan_to_num(self.regime_features[self.t], nan=0.0)
        features.extend(reg_feat.tolist())

        # Convert to numpy array and ensure no NaN values
        obs = np.array(features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0)
        
        return obs

    def step(self, actions):
        """
        actions: array in [-1, +1] of length n_pairs
        """
        # Clip actions to valid range
        pos = np.clip(actions, -1.0, 1.0).astype(np.float32)

        # 1) Compute per-pair z-scores at t and t+1 using price-based spreads
        zs_t = []
        zs_tp1 = []
        actual_returns = []
        for i, pair in enumerate(self.pairs):
            # Calculate z-scores using price-based spreads
            price_spread = self.spreads[pair]
            seg_t = price_spread[self.t - self.window : self.t]
            mean_t = np.nanmean(seg_t)
            std_t = np.nanstd(seg_t) + 1e-8
            z_t = (price_spread[self.t] - mean_t) / std_t
            zs_t.append(z_t)

            seg_tp1 = price_spread[self.t + 1 - self.window : self.t + 1]
            mean_tp1 = np.nanmean(seg_tp1)
            std_tp1 = np.nanstd(seg_tp1) + 1e-8
            z_tp1 = (price_spread[self.t + 1] - mean_tp1) / std_tp1
            zs_tp1.append(z_tp1)
            
            # Calculate actual returns using return-based spreads
            if pos[i] != 0:
                # Adjust index for return spreads (they start at index 1)
                ret_idx = self.t - self.return_start_idx
                if 0 <= ret_idx < len(self.return_spreads[pair]):
                    ret = self.return_spreads[pair][ret_idx]
                else:
                    ret = 0.0  # Default to 0 if index out of bounds
                actual_returns.append(pos[i] * ret)
            else:
                actual_returns.append(0.0)

        zs_t = np.array(zs_t, dtype=np.float32)
        zs_tp1 = np.array(zs_tp1, dtype=np.float32)
        actual_returns = np.array(actual_returns, dtype=np.float32)

        # 2) Compute raw PnL from actual returns
        raw_pnl = np.sum(actual_returns)

        # 3) Subtract transaction cost proportional to change in position
        trade_cost = self.cost * np.sum(np.abs(pos - self.prev_pos))
        raw_pnl -= trade_cost

        # 4) Calculate reward based on actual returns
        pnl = raw_pnl

        # 5) Update cumulative PnL and drawdown
        self.cum_pnl += pnl
        self.max_cum_pnl = max(self.max_cum_pnl, self.cum_pnl)
        drawdown = self.max_cum_pnl - self.cum_pnl

        # 6) Scale drawdown penalty by window_length
        dd_penalty = self.lambda_dd * (drawdown / float(self.window))

        # 7) Volatility penalty on actual returns
        vol_penalty = self.gamma_vol * np.abs(pnl)

        # 8) Macro multiplier based on SPX return at t+1
        spx_r = self.spx_ret[self.t + 1]
        if spx_r > 0:  # Bull market
            if pnl > 0:
                macro_multiplier = 1.2
            else:
                macro_multiplier = 0.7
        else:  # Bear market
            if pnl > 0:
                macro_multiplier = 1.5
            else:
                macro_multiplier = 0.9

        # 9) Small time‐step penalty if flat (all positions near zero)
        flat_penalty = 0.0
        if np.all(np.abs(pos) < 1e-3):
            flat_penalty = 0.05

        # 10) Market neutrality penalty (beta to SPX)
        # Estimate portfolio beta as the weighted sum of pair correlations to SPX
        pair_betas = np.array([
            self.correlations[pair][self.t-1] if (self.t-1) < len(self.correlations[pair]) else 0.0
            for pair in self.pairs
        ])
        if np.sum(np.abs(pos)) > 1e-8:
            portfolio_beta = np.dot(pos, pair_betas) / (np.sum(np.abs(pos)) + 1e-8)
        else:
            portfolio_beta = 0.0
        market_neutrality_penalty = self.lambda_beta * (portfolio_beta ** 2)

        # 11) Final reward
        reward = (pnl * macro_multiplier) - dd_penalty - vol_penalty - flat_penalty - market_neutrality_penalty

        # 12) Advance time index
        self.prev_pos = pos.copy()
        self.t += 1
        done = self.t >= (self.n_steps - 1)

        # 13) Construct next observation
        obs = self._get_obs()

        # 14) Info diagnostics
        info = {
            "pnl": float(pnl),
            "drawdown": float(drawdown),
            "spx_ret": float(spx_r),
            "position": pos.copy(),
            "actual_returns": actual_returns.tolist(),
            "portfolio_beta": float(portfolio_beta),
            "market_neutrality_penalty": float(market_neutrality_penalty)
        }

        return obs, float(reward), done, False, info

    def render(self):
        # (Optional) implement logging or plotting here if needed
        pass
