#!/usr/bin/env python3
import argparse
import numpy as np
from stable_baselines3 import PPO
from rl.environment import PairsEnv

def evaluate_model(model_path, prices_csv, pairs_csv, n_episodes, deterministic):
    model = PPO.load(model_path)
    env   = PairsEnv(prices_csv=prices_csv, pairs_csv=pairs_csv)

    rewards = []
    for i in range(n_episodes):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, _ = env.step(action)
            total += r
        rewards.append(total)
        print(f"Episode {i+1}/{n_episodes}: Reward = {total:.4f}")

    rewards = np.array(rewards)
    mean_r = rewards.mean()
    std_r  = rewards.std(ddof=1) if n_episodes>1 else float("nan")
    sharpe = mean_r/std_r if std_r and std_r!=0 else float("nan")
    print("\n--- Evaluation Summary ---")
    print(f"Episodes   : {n_episodes}")
    print(f"Avg Reward : {mean_r:.4f}")
    print(f"Std Reward : {std_r:.4f}")
    print(f"Sharpe     : {sharpe:.4f}")

def main():
    p = argparse.ArgumentParser("Evaluate RL pairs model")
    p.add_argument("--model",       required=True,
                   help="Path to trained model .zip")
    p.add_argument("--prices",      default="src/data/preprocessed/prices.csv")
    p.add_argument("--pairs",       default="src/pairing/pair_candidates.csv")
    p.add_argument("--episodes", "-n", type=int, default=5)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    evaluate_model(
        model_path=args.model,
        prices_csv=args.prices,
        pairs_csv=args.pairs,
        n_episodes=args.episodes,
        deterministic=args.deterministic
    )

if __name__ == "__main__":
    main()
