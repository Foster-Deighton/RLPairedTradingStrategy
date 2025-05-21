#!/usr/bin/env python3
"""
paper_trade.py

Loop (or one-off) paper-trade your SB3-driven pairs strategy via IB Gateway/TWS.
"""

import time
import argparse
import csv
from pathlib import Path
import logging
from collections import deque
import pickle
import random

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, util, MarketOrder
from stable_baselines3 import PPO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── ARGS ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Paper-trade RL pairs strategy via IB paper gateway"
)
parser.add_argument("--once",    action="store_true",
                    help="Run one iteration and exit")
parser.add_argument("--dry-run", action="store_true",
                    help="Print orders but do not submit")
parser.add_argument("--sleep",   type=int, default=5,
                    help="Seconds to sleep between loops (default: 86400s)")
parser.add_argument("--use-latest", action="store_true",
                    help="Use latest model instead of best performing model")
parser.add_argument("--iterations", type=int, default=None,
                    help="Number of iterations to run (default: run indefinitely)")
args = parser.parse_args()

# ── CONFIG ────────────────────────────────────────────────
IB_HOST       = '127.0.0.1'
IB_PORT       = 7497            # or 4002 if you use IB Gateway paper
CLIENT_ID     = random.randint(1, 10000)  # Random client ID to avoid conflicts
MODEL_DIR     = Path(__file__).parent.parent / "rl/models"
MODEL_PATH    = MODEL_DIR / "final_model.zip"
PAIRS_CSV     = Path(__file__).parent.parent / "pairing/pair_candidates.csv"
WINDOW        = 90              # days of history for z-score
TOTAL_CAPITAL = 100_000         # your starting notional
MAX_LEVERAGE  = 0.05            # 5% of capital per pair
LOG_CSV       = Path(__file__).parent / "paper_trade_log.csv"
STOP_LOSS_PCT = 0.02            # 2% stop loss per pair
TAKE_PROFIT_PCT = 0.03          # 3% take profit per pair

# Performance Tracking
PERFORMANCE_WINDOW = 100        # Number of iterations to track for performance
MIN_PERFORMANCE_IMPROVEMENT = 0.01  # Minimum improvement required to consider model better
PERFORMANCE_METRICS_FILE = MODEL_DIR / "performance_metrics.json"

# Adaptive Learning Parameters
INITIAL_LEARNING_RATE = 0.00005
MIN_LEARNING_RATE = 0.00001
MAX_LEARNING_RATE = 0.0001
LEARNING_RATE_ADJUSTMENT = 0.1  # How much to adjust learning rate by

# Testing Parameters
TEST_INTERVAL = 50              # Run tests every N iterations
TEST_WINDOW = 20               # Number of iterations to test over
MIN_TEST_WIN_RATE = 0.55       # Minimum win rate to consider model improving

# Learning Parameters
LEARNING_RATE = 0.00005         # Reduced learning rate for stability
SAVE_INTERVAL = 100             # Save model every N iterations
REPLAY_SIZE   = 5000            # Increased buffer size for better sampling
BATCH_SIZE    = 128             # Increased batch size for stability
MIN_REPLAY    = 500             # Increased minimum experiences before learning
BEST_MODEL_PATH = MODEL_DIR / "best_model.zip"
PERFORMANCE_HISTORY = MODEL_DIR / "performance_history.pkl"

# Multi-timeframe parameters
SHORT_TERM_WINDOW = 5           # 5-second window for short-term signals
MEDIUM_TERM_WINDOW = 20         # 20-day window for medium-term signals
SHORT_TERM_WEIGHT = 0.3         # Weight for short-term signals
MEDIUM_TERM_WEIGHT = 0.7        # Weight for medium-term signals

# Overfitting prevention
MAX_EPOCHS = 3                  # Maximum epochs per training iteration
EARLY_STOPPING_PATIENCE = 5     # Number of iterations without improvement before stopping
MIN_PERFORMANCE_THRESHOLD = 0.1 # Minimum performance improvement required
VALIDATION_SPLIT = 0.2          # Portion of data used for validation
MAX_DRAWDOWN_THRESHOLD = 0.05   # Maximum allowed drawdown before stopping training

class ExperienceReplay:
    def __init__(self, max_size=5000):
        self.buffer = deque(maxlen=max_size)
        self.short_term_buffer = deque(maxlen=1000)  # Separate buffer for short-term experiences
        self.medium_term_buffer = deque(maxlen=4000) # Separate buffer for medium-term experiences
    
    def add(self, obs, action, reward, next_obs, timeframe='medium'):
        """Add experience to appropriate buffer based on timeframe."""
        if timeframe == 'short':
            self.short_term_buffer.append((obs, action, reward, next_obs))
        else:
            self.medium_term_buffer.append((obs, action, reward, next_obs))
        self.buffer.append((obs, action, reward, next_obs))
    
    def sample(self, batch_size, timeframe='both'):
        """Sample experiences from specified timeframe(s)."""
        if timeframe == 'short':
            buffer = self.short_term_buffer
        elif timeframe == 'medium':
            buffer = self.medium_term_buffer
        else:
            buffer = self.buffer
            
        if len(buffer) < batch_size:
            return None
            
        indices = np.random.choice(len(buffer), batch_size, replace=False)
        return [buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class PerformanceTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.returns = deque(maxlen=window_size)
        self.win_rates = deque(maxlen=window_size)
        self.sharpe_ratios = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.pair_performance = {}  # Track performance by pair
        self.last_rewards = {}      # Track last reward by pair
        
    def update(self, reward, return_pct, win_rate, sharpe_ratio, learning_rate):
        self.rewards.append(reward)
        self.returns.append(return_pct)
        self.win_rates.append(win_rate)
        self.sharpe_ratios.append(sharpe_ratio)
        self.learning_rates.append(learning_rate)
        
    def update_pair_performance(self, pair_id, reward, spread_return):
        if pair_id not in self.pair_performance:
            self.pair_performance[pair_id] = deque(maxlen=self.window_size)
            self.last_rewards[pair_id] = 0
        
        # Calculate improvement from last reward
        improvement = reward - self.last_rewards[pair_id]
        self.pair_performance[pair_id].append({
            'reward': reward,
            'spread_return': spread_return,
            'improvement': improvement
        })
        self.last_rewards[pair_id] = reward
        
    def get_pair_metrics(self, pair_id):
        if pair_id not in self.pair_performance:
            return None
            
        performance = list(self.pair_performance[pair_id])
        if not performance:
            return None
            
        return {
            'avg_reward': np.mean([p['reward'] for p in performance]),
            'avg_spread_return': np.mean([p['spread_return'] for p in performance]),
            'avg_improvement': np.mean([p['improvement'] for p in performance]),
            'consistency': np.std([p['improvement'] for p in performance]) if len(performance) > 1 else 0
        }
    
    def get_metrics(self):
        return {
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'avg_return': np.mean(self.returns) if self.returns else 0,
            'avg_win_rate': np.mean(self.win_rates) if self.win_rates else 0,
            'avg_sharpe': np.mean(self.sharpe_ratios) if self.sharpe_ratios else 0,
            'current_learning_rate': self.learning_rates[-1] if self.learning_rates else INITIAL_LEARNING_RATE
        }

def adjust_learning_parameters(performance_tracker):
    """Adjust learning parameters based on performance."""
    metrics = performance_tracker.get_metrics()
    current_lr = metrics['current_learning_rate']
    
    if performance_tracker.is_improving():
        # If performance is improving, slightly increase learning rate
        new_lr = min(current_lr * (1 + LEARNING_RATE_ADJUSTMENT), MAX_LEARNING_RATE)
        logging.info(f"Performance improving, increasing learning rate to {new_lr:.6f}")
    else:
        # If performance is not improving, decrease learning rate
        new_lr = max(current_lr * (1 - LEARNING_RATE_ADJUSTMENT), MIN_LEARNING_RATE)
        logging.info(f"Performance not improving, decreasing learning rate to {new_lr:.6f}")
    
    return new_lr

def run_test_iteration():
    """Run a test iteration to evaluate model performance."""
    test_rewards = []
    test_returns = []
    test_win_rates = []
    
    for _ in range(TEST_WINDOW):
        run_iteration()
        test_rewards.append(returns_history[-1] if returns_history else 0)
        test_returns.append((net_liq / initial_equity - 1) * 100)
        
        # Calculate win rate for current positions
        current_positions = {p.contract.symbol: p for p in ib.positions()}
        winning_positions = sum(1 for p in current_positions.values() 
                              if p.position != 0 and 
                              float(p.avgCost) * (1 if p.position > 0 else -1) < 
                              float(price_data[p.contract.symbol].iloc[-1]))
        test_win_rates.append(winning_positions / len(current_positions) if current_positions else 0)
    
    return {
        'avg_reward': np.mean(test_rewards),
        'avg_return': np.mean(test_returns),
        'win_rate': np.mean(test_win_rates)
    }

def load_best_model():
    """Load the best performing model based on historical performance."""
    if not PERFORMANCE_HISTORY.exists():
        logging.info("No performance history found. Loading default model.")
        return PPO.load(str(MODEL_PATH))
    
    with open(PERFORMANCE_HISTORY, 'rb') as f:
        history = pickle.load(f)
    
    if not history:
        logging.info("Empty performance history. Loading default model.")
        return PPO.load(str(MODEL_PATH))
    
    # Find best model based on Sharpe ratio
    best_model_path = max(history.items(), key=lambda x: x[1]['sharpe'])[0]
    best_metrics = history[best_model_path]
    
    logging.info(f"Loading best performing model:")
    logging.info(f"  Path: {best_model_path}")
    logging.info(f"  Sharpe Ratio: {best_metrics['sharpe']:.2f}")
    logging.info(f"  Win Rate: {best_metrics['win_rate']:.2%}")
    logging.info(f"  Total Return: {best_metrics['total_return']:.2%}")
    
    return PPO.load(str(best_model_path))

def save_performance_metrics(model_path, metrics):
    """Save performance metrics for model evaluation."""
    if PERFORMANCE_HISTORY.exists():
        with open(PERFORMANCE_HISTORY, 'rb') as f:
            history = pickle.load(f)
    else:
        history = {}
    
    history[str(model_path)] = metrics
    
    # Log the new metrics
    logging.info(f"New model performance metrics:")
    logging.info(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    logging.info(f"  Win Rate: {metrics['win_rate']:.2%}")
    logging.info(f"  Total Return: {metrics['total_return']:.2%}")
    logging.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    with open(PERFORMANCE_HISTORY, 'wb') as f:
        pickle.dump(history, f)

def calculate_performance_metrics(returns):
    """Calculate key performance metrics with additional safeguards."""
    returns = np.array(returns)
    
    # Basic metrics
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
    max_drawdown = np.min(np.cumsum(returns)) if len(returns) > 0 else 0
    total_return = np.sum(returns)
    
    # Additional metrics for overfitting detection
    recent_returns = returns[-100:] if len(returns) >= 100 else returns
    recent_sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252) if len(recent_returns) > 0 else 0
    
    # Calculate consistency metrics
    return_std = np.std(returns) if len(returns) > 0 else 0
    return_skew = np.mean((returns - np.mean(returns))**3) / (np.std(returns)**3) if len(returns) > 0 else 0
    
    return {
        'sharpe': sharpe,
        'recent_sharpe': recent_sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'return_std': return_std,
        'return_skew': return_skew
    }

# ── INIT IB & RL MODEL ────────────────────────────────────
def initialize_trading():
    """Initialize or reset the trading environment."""
    global ib, model, replay_buffer, performance_tracker, iteration_count, returns_history, initial_equity, prev_equity
    
    # Disconnect if already connected
    try:
        ib.disconnect()
    except:
        pass
    
    # Initialize IB connection
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    
    # Reset performance tracking
    performance_tracker = PerformanceTracker(window_size=PERFORMANCE_WINDOW)
    replay_buffer = ExperienceReplay(max_size=REPLAY_SIZE)
    iteration_count = 0
    returns_history = []
    initial_equity = None
    prev_equity = None
    
    # Reset paper trade log
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp","net_liq","itteration_return_pct","cumulative_return_pct","reward"])
    logging.info("Paper trade log reset")
    
    # Load fresh model
    if args.use_latest:
        logging.info("Loading latest model as specified by --use-latest flag")
        model = PPO.load(str(MODEL_PATH))
    else:
        model = load_best_model()
    
    # Clear any existing positions
    for position in ib.positions():
        contract = position.contract
        if position.position > 0:
            ib.placeOrder(contract, MarketOrder('SELL', abs(position.position)))
        else:
            ib.placeOrder(contract, MarketOrder('BUY', abs(position.position)))
    
    logging.info("Trading environment initialized and reset")

# Initialize at start
initialize_trading()

# ── LOAD PAIRS & QUALIFY CONTRACTS ───────────────────────
pairs_df = pd.read_csv(PAIRS_CSV)
pair_list = []
contracts = {}
for _, row in pairs_df.iterrows():
    a, b = row["TickerA"], row["TickerB"]
    for sym in (a, b):
        if sym not in contracts:
            c = Stock(sym, "SMART", "USD")
            contracts[sym] = ib.qualifyContracts(c)[0]
    pair_list.append((a, b))

# ── SETUP LOG FILE ────────────────────────────────────────
# Note: Log file is now reset in initialize_trading()

def check_position_exits(ib, contracts, positions, price_data):
    """Check if any positions have hit stop loss or take profit and close them."""
    # Get all current positions
    current_positions = {p.contract.symbol: p for p in ib.positions()}
    
    for sym, contract in contracts.items():
        try:
            # Check if we have a position for this symbol
            if sym not in current_positions:
                continue
                
            pos = current_positions[sym]
            if pos.position == 0:
                continue
                
            entry_price = float(pos.avgCost)
            current_price = float(price_data[sym].iloc[-1])
            pnl_pct = (current_price - entry_price) / entry_price * (1 if pos.position > 0 else -1)
            
            if pnl_pct < -STOP_LOSS_PCT:
                logging.info(f"Stop loss triggered for {sym} at {pnl_pct:.2%}")
                ib.placeOrder(contract, MarketOrder('SELL' if pos.position > 0 else 'BUY', abs(pos.position)))
            elif pnl_pct > TAKE_PROFIT_PCT:
                logging.info(f"Take profit triggered for {sym} at {pnl_pct:.2%}")
                ib.placeOrder(contract, MarketOrder('SELL' if pos.position > 0 else 'BUY', abs(pos.position)))
        except Exception as e:
            logging.error(f"Error checking position exits for {sym}: {e}")

def calculate_reward(positions, price_data, iteration_return_pct):
    """Calculate reward based on iteration return percentage with extremely large rewards for gains."""
    try:
        # Convert percentage to decimal
        iteration_return = iteration_return_pct / 100.0
        
        # Make rewards extremely large and directly tied to iteration returns
        if iteration_return > 0:
            # Extremely large rewards for positive returns
            reward = iteration_return * 10000  # 10000x scaling for positive returns
        else:
            # Moderate penalties for negative returns
            reward = iteration_return * 2000  # 2000x scaling for negative returns

        # Add small exploration reward if no positions
        if not positions:
            reward = 0.1

        logging.info(f"Iteration return: {iteration_return:.2%}, Reward: {reward:.4f}")
        return reward
        
    except Exception as e:
        logging.error(f"Error calculating reward: {e}")
        return 0.0

def update_model(obs, reward, next_obs, action):
    """Update the model with new experience using replay buffer and overfitting safeguards."""
    global model, iteration_count, returns_history, learning_rate
    
    try:
        # Determine timeframe based on observation window
        timeframe = 'short' if len(obs[0]) <= SHORT_TERM_WINDOW else 'medium'
        
        # Add experience to replay buffer with timeframe
        replay_buffer.add(obs, action, reward, next_obs, timeframe)
        returns_history.append(reward)
        
        # Only start learning after collecting enough experiences
        if len(replay_buffer) >= MIN_REPLAY:
            # Sample batches for both timeframes
            short_batch = replay_buffer.sample(BATCH_SIZE, 'short')
            medium_batch = replay_buffer.sample(BATCH_SIZE, 'medium')
            
            if short_batch and medium_batch:
                # Combine batches with appropriate weights
                combined_batch = short_batch + medium_batch
                
                # Calculate validation split
                val_size = int(len(combined_batch) * VALIDATION_SPLIT)
                train_batch = combined_batch[:-val_size]
                val_batch = combined_batch[-val_size:]
                
                # Training loop with early stopping
                best_val_performance = float('-inf')
                patience_counter = 0
                
                for epoch in range(MAX_EPOCHS):
                    # Update model with training batch
                    model.learn(total_timesteps=1, 
                              tb_log_name="continuous_learning",
                              reset_num_timesteps=False)
                    
                    # Evaluate on validation batch
                    val_metrics = calculate_performance_metrics([r for _, _, r, _ in val_batch])
                    val_performance = val_metrics['sharpe']
                    
                    # Early stopping check
                    if val_performance > best_val_performance + MIN_PERFORMANCE_THRESHOLD:
                        best_val_performance = val_performance
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
        
        iteration_count += 1
        
        # Run tests periodically
        if iteration_count % TEST_INTERVAL == 0:
            test_results = run_test_iteration()
            logging.info(f"Test Results:")
            logging.info(f"  Average Reward: {test_results['avg_reward']:.4f}")
            logging.info(f"  Average Return: {test_results['avg_return']:.2%}")
            logging.info(f"  Win Rate: {test_results['win_rate']:.2%}")
            
            # Update performance tracker
            performance_tracker.update(
                reward=test_results['avg_reward'],
                return_pct=test_results['avg_return'],
                win_rate=test_results['win_rate'],
                sharpe_ratio=test_results['avg_reward'] / (np.std(returns_history[-TEST_WINDOW:]) + 1e-6),
                learning_rate=learning_rate
            )
            
            # Adjust learning parameters
            learning_rate = adjust_learning_parameters(performance_tracker)
            
            # Save model and metrics if performance is good
            if test_results['win_rate'] >= MIN_TEST_WIN_RATE:
                save_path = MODEL_DIR / f"model_iter_{iteration_count}.zip"
                model.save(str(save_path))
                
                # Calculate and save performance metrics
                metrics = calculate_performance_metrics(returns_history[-1000:])
                
                # Check for overfitting indicators
                if (metrics['recent_sharpe'] < metrics['sharpe'] * 0.5 or
                    metrics['max_drawdown'] < -MAX_DRAWDOWN_THRESHOLD or
                    abs(metrics['return_skew']) > 2.0):
                    logging.warning("Potential overfitting detected - skipping model save")
                    return
                
                save_performance_metrics(save_path, metrics)
                
                # Save best model if performance improved
                if metrics['sharpe'] > calculate_performance_metrics(returns_history[-2000:-1000])['sharpe']:
                    model.save(str(BEST_MODEL_PATH))
                    logging.info(f"New best model saved with Sharpe ratio: {metrics['sharpe']:.2f}")
                
                logging.info(f"Saved model and metrics - Sharpe: {metrics['sharpe']:.2f}, Win Rate: {metrics['win_rate']:.2%}")
            
    except Exception as e:
        logging.error(f"Error updating model: {e}")

def run_iteration():
    global initial_equity, prev_equity, model, returns_history, learning_rate

    logging.info("Starting iteration...")

    # 1) fetch net liquidation and available funds
    av = {v.tag: float(v.value) for v in ib.accountValues()
          if v.tag in ("NetLiquidation","TotalCashValue","AvailableFunds")}
    net_liq = av.get("NetLiquidation", av.get("TotalCashValue", np.nan))
    available_funds = av.get("AvailableFunds", net_liq)
    
    if initial_equity is None:
        initial_equity = net_liq
        prev_equity = net_liq
    
    # Calculate iteration and cumulative returns
    iteration_return_pct = ((net_liq / prev_equity) - 1) * 100
    cumulative_return_pct = ((net_liq / initial_equity) - 1) * 100
    
    # Update previous equity for next iteration
    prev_equity = net_liq
    
    logging.info(f"Net Liquidation: {net_liq}")
    logging.info(f"Available Funds: {available_funds}")
    logging.info(f"Iteration Return: {iteration_return_pct:.2f}%")
    logging.info(f"Cumulative Return: {cumulative_return_pct:.2f}%")

    # 2) fetch last WINDOW+1 daily bars for each symbol
    price_data = {}
    for sym, contract in contracts.items():
        logging.info(f"Fetching data for {sym}...")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=f'{WINDOW+1} D',
            barSizeSetting='1 day',
            whatToShow='ADJUSTED_LAST',
            useRTH=True
        )
        df = util.df(bars)
        price_data[sym] = df.set_index('date')['close']
        logging.info(f"Data fetched for {sym}.")

    # 3) build the z-score observation window
    obs_segments = []
    for a, b in pair_list:
        sa = price_data.get(a)
        sb = price_data.get(b)
        if sa is None or sb is None or len(sa) < WINDOW+1:
            logging.warning(f"Insufficient data for pair {a},{b}; skipping iteration.")
            obs_segments = []
            break

        spread = sa.values - sb.values
        mu, sigma = spread.mean(), spread.std()
        zs = (spread - mu) / sigma
        obs_segments.append(zs[-WINDOW:])
    if not obs_segments:
        logging.warning("Insufficient data; skipping iteration.")
        return

    # Check position exits before making new decisions
    check_position_exits(ib, contracts, ib.positions(), price_data)

    obs = np.concatenate(obs_segments)[None,:].astype(np.float32)
    logging.info("Observation window built.")

    # 4) predict signals {0,1,2} → {0,+1,–1}
    acts = model.predict(obs, deterministic=True)[0]
    acts = acts.flatten()
    sigs = np.where(acts==1, +1, np.where(acts==2, -1, 0))
    sigs = sigs.flatten()
    sigs = [float(sig) for sig in sigs]
    logging.info(f"Signals generated: {sigs}")

    # Store current observation for next iteration
    next_obs = obs.copy()

    # 5) size and send orders
    total_notional = 0
    order_specs = []
    for (a, b), sig in zip(pair_list, sigs):
        # Scale position size by z-score magnitude
        z_score = float(obs[0][-1])  # Get latest z-score
        position_scale = min(abs(z_score) / 1.0, 1.0)  # More aggressive scaling (changed from 2.0)
        target_notional = float(TOTAL_CAPITAL * MAX_LEVERAGE * sig * position_scale)
        val_a = float(0.5 * target_notional)
        val_b = float(-0.5 * target_notional)

        # current position values
        try:
            pa = ib.position(contracts[a])
            pb = ib.position(contracts[b])
            cur_a = float(pa.position) * float(pa.avgCost)
            cur_b = float(pb.position) * float(pb.avgCost)
        except:
            cur_a = cur_b = 0.0

        price_a = float(price_data[a].iloc[-1])
        price_b = float(price_data[b].iloc[-1])
        qty_a = int((val_a - cur_a) / price_a)
        qty_b = int((val_b - cur_b) / price_b)

        # Apply 500 share limit while maintaining proportions
        max_shares = 500
        if abs(qty_a) > max_shares or abs(qty_b) > max_shares:
            scale_factor = max_shares / max(abs(qty_a), abs(qty_b))
            qty_a = int(qty_a * scale_factor)
            qty_b = int(qty_b * scale_factor)
            logging.info(f"Position sizes scaled down to respect 500 share limit. Scale factor: {scale_factor:.2f}")

        # Calculate notional for this pair
        notional_a = abs(qty_a * price_a)
        notional_b = abs(qty_b * price_b)
        pair_notional = notional_a + notional_b
        order_specs.append({
            'a': a, 'b': b, 'sig': sig, 'qty_a': qty_a, 'qty_b': qty_b,
            'price_a': price_a, 'price_b': price_b, 'notional': pair_notional
        })
        total_notional += pair_notional

    # If total notional exceeds available funds, scale all orders down
    if total_notional > available_funds:
        scale_factor = available_funds / total_notional
        logging.warning(f"Scaling down all orders by {scale_factor:.2f} to fit available funds.")
        for spec in order_specs:
            spec['qty_a'] = int(spec['qty_a'] * scale_factor)
            spec['qty_b'] = int(spec['qty_b'] * scale_factor)

    # Place orders
    min_position_value = TOTAL_CAPITAL * MAX_LEVERAGE * 0.1
    for spec in order_specs:
        a, b, sig = spec['a'], spec['b'], spec['sig']
        qty_a, qty_b = spec['qty_a'], spec['qty_b']
        price_a, price_b = spec['price_a'], spec['price_b']
        if abs(qty_a * price_a) >= min_position_value or abs(qty_b * price_b) >= min_position_value:
            logging.info(f"Signal ({a},{b})={sig:+.0f} → qtyA={qty_a}, qtyB={qty_b}")
            if not args.dry_run:
                if qty_a:
                    logging.info(f"Placing order for {a}: {'BUY' if qty_a>0 else 'SELL'} {abs(qty_a)}")
                    ib.placeOrder(contracts[a],
                                  MarketOrder('BUY' if qty_a>0 else 'SELL', abs(qty_a)))
                if qty_b:
                    logging.info(f"Placing order for {b}: {'BUY' if qty_b>0 else 'SELL'} {abs(qty_b)}")
                    ib.placeOrder(contracts[b],
                                  MarketOrder('BUY' if qty_b>0 else 'SELL', abs(qty_b)))
        else:
            logging.info(f"Signal ({a},{b})={sig:+.0f} too small to trade (min value: {min_position_value:.2f})")

    # 6) log equity & returns
    logging.info(f"Equity=${net_liq:,.2f}  Iteration Return={iteration_return_pct:.2f}%  Cumulative Return={cumulative_return_pct:.2f}%")
    
    # Calculate reward and update model
    reward = calculate_reward(ib.positions(), price_data, iteration_return_pct)
    update_model(obs, reward, next_obs, acts)
    
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{net_liq:.2f}",
            f"{iteration_return_pct:.4f}",
            f"{cumulative_return_pct:.4f}",
            f"{reward:.4f}"
        ])

if __name__ == "__main__":
    try:
        iteration_count = 0
        while True:
            if args.iterations is not None and iteration_count >= args.iterations:
                logging.info(f"Completed {args.iterations} iterations as specified")
                break
            run_iteration()
            iteration_count += 1
            if args.once:
                print("🔚 --once flag set; exiting.")
                break
            time.sleep(args.sleep)
    finally:
        ib.disconnect()
