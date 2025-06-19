#!/usr/bin/env python3
"""
Walk‐forward RL training with Sortino‐penalty, regime overlay, macro features, and ensemble across folds.

Run from src/rl:
    python train.py --prices path/to/prices.csv --pairs path/to/pair_candidates.csv \
         --timesteps 300000 --eval-pct 0.2 \
         --beta 0.1 --trend-weight 0.5 --sortino-window 252 --sortino-lambda 5.0 \
         --lambda-dd 0.5 --gamma-vol 0.1
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CallbackList
)
from environment import PairsEnv


def parse_args():
    p = argparse.ArgumentParser("Walk‐forward RL training with macro features and ensemble")
    p.add_argument("--prices",        required=False,
                   default="../data/preprocessed/prices.csv",
                   help="Full preprocessed prices CSV")
    p.add_argument("--pairs",         required=False,
                   default="../pairing/pair_candidates.csv",
                   help="Pair candidates CSV")
    p.add_argument("--model-dir",     default="models", help="Directory to save fold models")
    p.add_argument("--timesteps",     type=int,   default=300_000, help="PPO timesteps per fold")
    p.add_argument("--eval-episodes", type=int,   default=5,      help="Eval episodes per fold")
    p.add_argument("--eval-pct",      type=float, default=0.2,    help="Fraction reserved for OOS eval")

    # Regime overlay
    p.add_argument("--beta",          type=float, default=0.2,  help="SPY return weight (beta_spy)")
    p.add_argument("--trend-weight",  type=float, default=0.7,  help="SPY trend position sizing")

    # Sortino penalty (currently not used directly in PairsEnv, included for future extension)
    p.add_argument("--sortino-window", type=int,   default=252, help="Lookback for downside dev")
    p.add_argument("--sortino-lambda", type=float, default=3.0, help="Sortino penalty lambda")

    # Macro feature penalties
    p.add_argument("--lambda-dd",    type=float, default=0.5, help="Drawdown penalty weight")
    p.add_argument("--gamma-vol",    type=float, default=0.1, help="Volatility penalty weight")

    # PPO hyperparameters (adjusted)
    p.add_argument("--lr",            type=float, default=3e-5,  help="Learning rate")
    p.add_argument("--n-steps",       type=int,   default=1024,  help="Steps per env update")
    p.add_argument("--batch-size",    type=int,   default=128,   help="Minibatch size")
    p.add_argument("--gamma",         type=float, default=0.99,  help="Discount factor")
    p.add_argument("--gae-lambda",    type=float, default=0.95,  help="GAE lambda")
    p.add_argument("--clip-range",    type=float, default=0.1,   help="PPO clip range")
    p.add_argument("--ent-coef",      type=float, default=0.005, help="Entropy coefficient")
    p.add_argument("--vf-coef",       type=float, default=0.5,   help="Value function coefficient")
    p.add_argument("--max-grad-norm", type=float, default=0.3,   help="Max gradient norm")
    p.add_argument("--weight-decay",  type=float, default=1e-5,  help="Weight decay (AdamW)")
    p.add_argument("--h1",            type=int,   default=64,    help="Hidden layer1 size")
    p.add_argument("--h2",            type=int,   default=64,    help="Hidden layer2 size")
    return p.parse_args()


def slice_fold(prices_csv: str, dates: pd.DatetimeIndex, start_idx: int, end_idx: int) -> str:
    df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    df = df.sort_index().loc[dates[start_idx]:dates[end_idx]]
    if df.empty:
        raise ValueError(f"Slice {dates[start_idx].date()}–{dates[end_idx].date()} empty")
    out = Path(tempfile.gettempdir()) / f"prices_{dates[start_idx].date()}_{dates[end_idx].date()}.csv"
    df.to_csv(out)
    return str(out)


def main():
    args = parse_args()
    base = Path(args.model_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Load and validate data
    full = pd.read_csv(args.prices, index_col=0, parse_dates=True).sort_index()
    dates = full.index.normalize().unique()
    n = len(dates)

    if n < 270:
        raise ValueError(f"Insufficient data points: {n} < 270 (minimum required)")

    eval_len = max(1, int(n * args.eval_pct))
    train_len = n - eval_len
    step = eval_len

    metrics = []
    fold_dirs = []

    last_start = n - (train_len + eval_len)
    if last_start < 0:
        last_start = 0

    for fold, start in enumerate(range(0, last_start + 1, step)):
        te_idx = start + train_len - 1
        ee_idx = te_idx + eval_len
        if ee_idx >= n:
            break

        ts, te = dates[start], dates[te_idx]
        es, ee = dates[te_idx + 1], dates[ee_idx]
        print(f"Fold {fold}: TRAIN {ts.date()}–{te.date()} | EVAL {es.date()}–{ee.date()}")

        train_csv = slice_fold(args.prices, dates, start, te_idx)
        eval_csv = slice_fold(args.prices, dates, te_idx + 1, ee_idx)

        fold_dir = base / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        fold_dirs.append(fold_dir)

        def make_env(csv_path, monitor_file):
            return Monitor(
                PairsEnv(
                    prices_csv=csv_path,
                    pairs_csv=args.pairs,
                    regime_csv="../data/preprocessed/regime.csv",
                    spx_csv="../data/preprocessed/SPX_returns.csv",
                    window_length=90,
                    trading_cost=0.0005,
                    beta_spy=args.beta
                ),
                filename=str(monitor_file)
            )

        # Use 4 parallel envs for training
        train_env = DummyVecEnv([
            (lambda csv=train_csv, mon=fold_dir / f'train_monitor_{i}.csv': make_env(csv, mon))
            for i in range(4)
        ])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        # Use 2 parallel envs for evaluation
        eval_env = DummyVecEnv([
            (lambda csv=eval_csv, mon=fold_dir / f'eval_monitor_{i}.csv': make_env(csv, mon))
            for i in range(2)
        ])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        eval_env.training = False
        eval_env.norm_reward = False

        # callbacks
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(fold_dir),
            log_path=str(fold_dir),
            eval_freq=50_000,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            callback_after_eval=stop_cb
        )

        policy_kwargs = dict(net_arch=[args.h1, args.h2])
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=lambda f: args.lr * (1 - 0.5 * f),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=lambda f: args.clip_range * (1 - 0.5 * f),
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(fold_dir / 'tb')
        )
        model.learn(total_timesteps=args.timesteps, callback=CallbackList([eval_cb]))
        model.save(str(fold_dir / 'final_model'))

        # parse evaluation metrics
        try:
            eval_files = list(fold_dir.glob('eval_monitor_*.csv'))
            if eval_files:
                all_rewards = []
                for eval_file in eval_files:
                    with open(eval_file, 'r') as f:
                        lines = f.readlines()
                    # Skip any line starting with '#' or 'r,' (header)
                    data_lines = [l for l in lines if not l.startswith('#') and not l.startswith('r,')]
                    if data_lines:
                        rewards = []
                        for line in data_lines:
                            try:
                                rewards.append(float(line.split(',')[0]))
                            except ValueError:
                                continue
                        all_rewards.extend(rewards)
                mean_oos = np.mean(all_rewards) if all_rewards else float('nan')
            else:
                print(f"Warning: No evaluation files found in {fold_dir}")
                mean_oos = float('nan')
        except Exception as e:
            print(f"Warning: Could not read evaluation metrics: {e}")
            mean_oos = float('nan')

        metrics.append({"fold": fold, "mean_oos_reward": mean_oos})

    # summary
    summary = pd.DataFrame(metrics)
    summary['overall_mean_oos'] = summary['mean_oos_reward'].mean()
    summary.to_csv(base / 'walkforward_summary.csv', index=False)
    print(summary)

    # ensemble on last eval window
    if n > eval_len:
        last_eval_csv = slice_fold(args.prices, dates, n - eval_len, n - 1)
        env = DummyVecEnv([
            (lambda csv=last_eval_csv, mon=Path(tempfile.gettempdir()) / f'ensemble_{i}.csv': make_env(csv, mon))
            for i in range(2)
        ])
        obs = env.reset()
        done = [False] * len(env.envs)
        models = [PPO.load(str(d / 'final_model')) for d in fold_dirs]
        total_reward = 0.0

        while not all(done):
            actions = [m.predict(obs, deterministic=True)[0] for m in models]
            if isinstance(actions[0], np.ndarray):
                action = np.sign(np.mean(actions, axis=0))
            else:
                action = max(set(actions), key=actions.count)
            obs, reward, done, _ = env.step(action)
            total_reward += np.mean(reward)  # Average reward across parallel envs
        print(f"Ensemble reward on last window: {total_reward}")


if __name__ == "__main__":
    main()
