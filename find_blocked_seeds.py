from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from environment.microrobot_env import MicrorobotEnv, load_config
from evaluate import run_episode
from stable_baselines3 import PPO


def find_seeds(n_trials: int = 200, start_seed: int = 5000, dense: bool = False):
    cfg_path = "config/config_dense.yaml" if dense else "config/config.yaml"
    cfg = load_config(cfg_path)
    env_h = MicrorobotEnv(config=cfg, enable_obstacles=True)
    env_p = MicrorobotEnv(config=cfg, enable_obstacles=True)
    model = PPO.load("models/checkpoints/exp02_cont_270832_steps.zip")

    rows = []
    found = []
    for i in range(n_trials):
        seed = start_seed + i
        r_h = run_episode(env_h, "heading", deterministic=True, seed=seed)
        r_p = run_episode(env_p, model, deterministic=True, seed=seed)
        rows.append({
            "seed": seed,
            "heading_success": r_h["success"],
            "heading_collision": r_h["collision"],
            "heading_return": r_h["total_reward"],
            "ppo_success": r_p["success"],
            "ppo_collision": r_p["collision"],
            "ppo_return": r_p["total_reward"],
        })
        if (not r_h["success"]) and r_p["success"]:
            found.append(seed)
            print(f"  FOUND seed={seed}: heading FAIL (collision={r_h['collision']}), PPO SUCCESS (steps={r_p['steps']})")
        if len(found) >= 4:
            break

    env_h.close()
    env_p.close()
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"results/tables/blocked_seed_search{'_dense' if dense else ''}.csv", index=False)
    print(f"\nFound seeds where PPO succeeds, heading fails: {found}")
    return found


if __name__ == "__main__":
    print("Searching standard obstacles (n=3)...")
    std = find_seeds(200, 5000, dense=False)
    print("\nSearching dense obstacles (n=6)...")
    dns = find_seeds(200, 6000, dense=True)
    print("\nFinal std seeds:", std)
    print("Final dense seeds:", dns)
