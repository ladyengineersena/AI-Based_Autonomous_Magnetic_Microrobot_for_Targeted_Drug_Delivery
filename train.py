"""Train a PPO controller for magnetic microrobot navigation."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.ppo_agent import PPOAgent
from environment.microrobot_env import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the microrobot environment")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--obstacles", action="store_true", help="Train with random rectangular/circular obstacles")
    parser.add_argument("--output", default="models/checkpoints/ppo_microrobot_final")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    agent = PPOAgent(config=config)
    agent.build(enable_obstacles=True if args.obstacles else False, seed=args.seed)
    agent.train(timesteps=args.timesteps)
    output = Path(args.output)
    agent.save(output)
    print(f"Saved PPO model to {output}.zip")


if __name__ == "__main__":
    main()
