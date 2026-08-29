"""Stable-Baselines3 PPO wrapper used by train.py and evaluate.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.microrobot_env import MicrorobotEnv, load_config


def make_env(config: dict[str, Any], enable_obstacles: bool | None = None, seed: int = 0):
    def _init():
        env = MicrorobotEnv(config=config, enable_obstacles=enable_obstacles)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


class PPOAgent:
    def __init__(self, config: dict[str, Any] | None = None, config_path: str | Path | None = None) -> None:
        self.config = config or load_config(config_path)
        self.model: PPO | None = None

    def build(self, enable_obstacles: bool | None = None, seed: int = 42) -> PPO:
        ppo_cfg = self.config["ppo"]
        env = DummyVecEnv([make_env(self.config, enable_obstacles=enable_obstacles, seed=seed)])
        self.model = PPO(
            ppo_cfg["policy"],
            env,
            learning_rate=float(ppo_cfg["learning_rate"]),
            n_steps=int(ppo_cfg["n_steps"]),
            batch_size=int(ppo_cfg["batch_size"]),
            n_epochs=int(ppo_cfg["n_epochs"]),
            gamma=float(ppo_cfg["gamma"]),
            gae_lambda=float(ppo_cfg["gae_lambda"]),
            clip_range=float(ppo_cfg["clip_range"]),
            ent_coef=float(ppo_cfg["ent_coef"]),
            vf_coef=float(ppo_cfg["vf_coef"]),
            max_grad_norm=float(ppo_cfg["max_grad_norm"]),
            policy_kwargs={"net_arch": list(ppo_cfg["net_arch"])},
            verbose=1,
            seed=seed,
            tensorboard_log="results/logs",
        )
        return self.model

    def train(self, timesteps: int | None = None, checkpoint_dir: str = "models/checkpoints") -> PPO:
        if self.model is None:
            self.build()
        assert self.model is not None
        ppo_cfg = self.config["ppo"]
        total = int(timesteps if timesteps is not None else ppo_cfg["total_timesteps"])
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=int(ppo_cfg["save_freq"]),
            save_path=checkpoint_dir,
            name_prefix="ppo_microrobot",
        )
        self.model.learn(total_timesteps=total, callback=callback, progress_bar=False)
        return self.model

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: str | Path, env=None) -> PPO:
        self.model = PPO.load(str(path), env=env)
        return self.model
