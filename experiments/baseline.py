"""Experiment 0 / baseline: random magnetic headings vs greedy field alignment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import run_episode, summarize
from environment.microrobot_env import MicrorobotEnv, load_config


def main() -> None:
    config = load_config()
    env = MicrorobotEnv(config=config, enable_obstacles=False)
    for name in ("random", "heading"):
        rows = []
        for episode in range(50):
            rows.append(run_episode(env, name, seed=1000 + episode))
        metrics = summarize(rows)
        print(f"\nBaseline: {name}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    env.close()


if __name__ == "__main__":
    main()
