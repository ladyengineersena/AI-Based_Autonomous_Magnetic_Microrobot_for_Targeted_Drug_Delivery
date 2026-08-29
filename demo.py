"""Save a single-episode trajectory figure (no interactive window required)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from environment.microrobot_env import MicrorobotEnv, load_config
from evaluate import greedy_heading_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--policy", choices=["heading", "random", "ppo"], default="heading")
    parser.add_argument("--model", default="models/checkpoints/ppo_microrobot_final.zip")
    parser.add_argument("--obstacles", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results/figures/demo_trajectory.png")
    args = parser.parse_args()

    config = load_config(args.config)
    env = MicrorobotEnv(config=config, enable_obstacles=args.obstacles)
    obs, _ = env.reset(seed=args.seed)
    policy = None
    if args.policy == "ppo":
        from stable_baselines3 import PPO

        policy = PPO.load(args.model)

    xs, ys = [env.robot.x], [env.robot.y]
    done = truncated = False
    while not (done or truncated):
        if args.policy == "random":
            action = env.action_space.sample()
        elif args.policy == "heading":
            action = greedy_heading_action(env)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(int(action))
        xs.append(env.robot.x)
        ys.append(env.robot.y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.world_size)
    ax.set_ylim(0, env.world_size)
    ax.set_aspect("equal")
    ax.set_title(f"Microrobot trajectory ({args.policy})")
    for obstacle in env.world.obstacles:
        if obstacle.kind == "circle":
            ax.add_patch(Circle((obstacle.x, obstacle.y), obstacle.radius, color="#4b5563"))
        else:
            ax.add_patch(Rectangle((obstacle.x, obstacle.y), obstacle.width, obstacle.height, color="#4b5563"))
    ax.add_patch(Circle((env.target[0], env.target[1]), env.world.target_radius, color="#f59e0b", label="Target"))
    ax.plot(xs, ys, color="#2563eb", linewidth=2, label="Path")
    ax.scatter(xs[0], ys[0], color="#16a34a", s=40, zorder=5, label="Start")
    ax.scatter(xs[-1], ys[-1], color="#2563eb", s=40, zorder=5, label="End")
    ax.legend(loc="upper right")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    env.close()
    print(f"Saved {output} success={info['success']} steps={info['steps']} distance={info['distance']:.4f}")


if __name__ == "__main__":
    main()
