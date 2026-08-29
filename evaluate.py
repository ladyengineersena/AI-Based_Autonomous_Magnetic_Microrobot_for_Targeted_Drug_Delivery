"""Evaluate a trained policy or a baseline controller."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from environment.microrobot_env import MicrorobotEnv, load_config


def greedy_heading_action(env: MicrorobotEnv) -> int:
    """Heuristic baseline: point the magnetic field at the target."""
    dx = env.target[0] - env.robot.x
    dy = env.target[1] - env.robot.y
    angle = float(np.arctan2(dy, dx))
    candidates = np.linspace(0.0, 2.0 * np.pi, env.action_space.n, endpoint=False)
    wrapped = (candidates - angle + np.pi) % (2.0 * np.pi) - np.pi
    return int(np.argmin(np.abs(wrapped)))


def run_episode(
    env: MicrorobotEnv,
    policy,
    deterministic: bool = True,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed, options=options)
    start = np.array([env.robot.x, env.robot.y], dtype=np.float64)
    target = env.target.copy()
    done = False
    truncated = False
    total_reward = 0.0
    while not (done or truncated):
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "heading":
            action = greedy_heading_action(env)
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += float(reward)

    optimal = float(np.linalg.norm(target - start))
    actual = float(info["path_length"]) if info["success"] else float("nan")
    efficiency = (optimal / actual) if info["success"] and actual > 1e-9 else float("nan")
    return {
        "success": int(info["success"]),
        "collision": int(info["collision"]),
        "time": info["time"],
        "steps": info["steps"],
        "path_length": info["path_length"],
        "final_error": info["distance"],
        "total_reward": total_reward,
        "optimal_path": optimal,
        "path_efficiency": efficiency,
        "control_effort": info["steps"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    successes = frame[frame["success"] == 1]
    return {
        "n_episodes": float(len(frame)),
        "success_rate": 100.0 * float(frame["success"].mean()),
        "collision_rate": float(frame["collision"].mean()),
        "avg_time": float(successes["time"].mean()) if len(successes) else float("nan"),
        "avg_path_length": float(successes["path_length"].mean()) if len(successes) else float("nan"),
        "avg_path_efficiency": float(successes["path_efficiency"].mean()) if len(successes) else float("nan"),
        "avg_final_error": float(frame["final_error"].mean()),
        "avg_control_effort": float(frame["control_effort"].mean()),
        "avg_return": float(frame["total_reward"].mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate microrobot controllers")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default="models/checkpoints/ppo_microrobot_final.zip")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--policy", choices=["ppo", "random", "heading"], default="ppo")
    parser.add_argument("--obstacles", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--csv", default="results/tables/evaluation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    n_episodes = args.episodes or int(config["evaluation"]["n_episodes"])
    env = MicrorobotEnv(config=config, enable_obstacles=args.obstacles)

    policy: Any
    if args.policy == "ppo":
        from stable_baselines3 import PPO

        policy = PPO.load(args.model)
    else:
        policy = args.policy

    rows = []
    for episode in range(n_episodes):
        rows.append(
            run_episode(
                env,
                policy,
                deterministic=bool(config["evaluation"]["deterministic"]),
                seed=args.seed + episode,
            )
        )

    env.close()
    metrics = summarize(rows)
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    summary_path = out.with_name(out.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("Evaluation summary")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"Wrote {out} and {summary_path}")


if __name__ == "__main__":
    main()
