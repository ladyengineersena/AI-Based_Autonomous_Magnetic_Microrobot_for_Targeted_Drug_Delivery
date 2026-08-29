"""Experiment 3: generalization across random start and goal poses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.microrobot_env import MicrorobotEnv, load_config
from evaluate import run_episode, summarize
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "checkpoints" / "ppo_microrobot_final.zip"))
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    config = load_config()
    model = PPO.load(args.model)
    env = MicrorobotEnv(config=config, enable_obstacles=False)
    rows = []
    for episode in range(args.episodes):
        rows.append(run_episode(env, model, seed=4000 + episode))
    env.close()
    print("\nExperiment 3 — unseen random starts")
    for key, value in summarize(rows).items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
