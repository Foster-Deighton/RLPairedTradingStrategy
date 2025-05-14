#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from environment import PairsEnv

def parse_args():
    # Get the project root directory (2 levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    
    p = argparse.ArgumentParser("Train RL agent for pairs trading")
    p.add_argument("--prices",    default=str(project_root / "src/data/preprocessed/prices.csv"))
    p.add_argument("--pairs",     default=str(project_root / "src/pairing/pair_candidates.csv"))
    p.add_argument("--model-dir", default=str(project_root / "src/rl/models"))
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # training & evaluation envs (for callbacks)
    train_env = DummyVecEnv([
        lambda: Monitor(
            PairsEnv(args.prices, args.pairs),
            filename=os.path.join(args.model_dir, "train_monitor.csv")
        )
    ])
    eval_env = DummyVecEnv([
        lambda: Monitor(
            PairsEnv(args.prices, args.pairs),
            filename=os.path.join(args.model_dir, "eval_monitor.csv")
        )
    ])

    # EvalCallback to save best model by evaluation performance
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.model_dir,
        eval_freq=20_000,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(args.model_dir, "tb_logs")
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    # final save
    model.save(os.path.join(args.model_dir, "final_model"))
    print("Training complete. Models & logs in", args.model_dir)

if __name__ == "__main__":
    main()
