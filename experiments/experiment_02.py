"""Experiment 2: PPO with static obstacles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ppo_agent import PPOAgent
from environment.microrobot_env import MicrorobotEnv, load_config
from evaluate import run_episode, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=120000)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    config = load_config()
    agent = PPOAgent(config=config)
    agent.build(enable_obstacles=True, seed=42)
    agent.train(timesteps=args.timesteps)
    agent.save(ROOT / "models" / "checkpoints" / "exp02_ppo_obstacles")

    env = MicrorobotEnv(config=config, enable_obstacles=True)
    rows = []
    for episode in range(args.episodes):
        rows.append(run_episode(env, agent.model, seed=3000 + episode))
    env.close()
    print("\nExperiment 2 — PPO with obstacles")
    for key, value in summarize(rows).items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
