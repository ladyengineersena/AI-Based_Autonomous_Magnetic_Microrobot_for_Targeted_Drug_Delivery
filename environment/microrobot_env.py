"""Gymnasium environment for 2D magnetic microrobot navigation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from environment.physics import (
    N_DISCRETE_ACTIONS,
    RobotState,
    action_to_field_angle,
    integrate_step,
    wrap_angle,
)
from simulation.world import SimulationWorld


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class MicrorobotEnv(gym.Env):
    """Navigate a magnetic microrobot to a target in a planar workspace.

    Observation: [x, y, theta / pi, dx, dy] with positions in [0, 1].
    Action: discrete magnetic field heading (8 compass directions).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        render_mode: str | None = None,
        enable_obstacles: bool | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_config(config_path)
        sim = self.config["simulation"]
        phys = self.config["physics"]
        env_cfg = self.config["environment"]
        reward_cfg = self.config["reward"]

        self.world_size = float(sim["world_size"])
        self.dt = float(sim["dt"])
        self.max_steps = int(sim["max_steps"])
        self.robot_radius = float(sim["robot_radius"])
        self.goal_threshold = float(sim["goal_threshold"])
        self.v_max = float(phys["v_max"])
        self.k_align = float(phys["k_align"])
        self.max_omega = float(phys["max_omega"])
        self.b_magnitude = float(phys["b_magnitude"])
        self.min_start_goal_distance = float(env_cfg["min_start_goal_distance"])
        self.enable_obstacles = (
            bool(env_cfg["enable_obstacles"]) if enable_obstacles is None else enable_obstacles
        )
        self.n_obstacles = int(env_cfg["n_obstacles"])
        self.obstacle_min_size = float(env_cfg["obstacle_min_size"])
        self.obstacle_max_size = float(env_cfg["obstacle_max_size"])
        self.goal_bonus = float(reward_cfg["goal_bonus"])
        self.collision_penalty = float(reward_cfg["collision_penalty"])
        self.step_penalty = float(reward_cfg["step_penalty"])
        self.distance_scale = float(reward_cfg["distance_scale"])

        self.world = SimulationWorld(
            world_size=self.world_size,
            robot_radius=self.robot_radius,
            target_radius=float(sim["target_radius"]),
            goal_threshold=self.goal_threshold,
            wall_collision=bool(env_cfg["wall_collision"]),
        )

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(int(self.config["action"]["n_discrete"]))
        if self.action_space.n != N_DISCRETE_ACTIONS:
            raise ValueError("Config n_discrete must match physics.N_DISCRETE_ACTIONS")

        self.render_mode = render_mode
        self._renderer = None
        self.robot = RobotState(0.1, 0.1, 0.0)
        self.target = np.array([0.8, 0.8], dtype=np.float64)
        self._steps = 0
        self._prev_distance = 0.0
        self.path_length = 0.0
        self.last_field_angle = 0.0
        self.last_info: dict[str, Any] = {}

    def _observe(self) -> np.ndarray:
        dx = self.target[0] - self.robot.x
        dy = self.target[1] - self.robot.y
        obs = np.array(
            [
                self.robot.x / self.world_size,
                self.robot.y / self.world_size,
                wrap_angle(self.robot.theta) / np.pi,
                dx / self.world_size,
                dy / self.world_size,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}
        rng = np.random.default_rng(seed if seed is not None else self.np_random.integers(0, 2**31 - 1))

        if self.enable_obstacles:
            self.world.spawn_obstacles(
                rng, self.n_obstacles, self.obstacle_min_size, self.obstacle_max_size
            )
        else:
            self.world.obstacles = []

        robot_xy = options.get("robot")
        target_xy = options.get("target")
        if robot_xy is None:
            robot_xy = self.world.sample_free_point(rng, self.robot_radius)
        else:
            robot_xy = np.asarray(robot_xy, dtype=np.float64)
        if target_xy is None:
            target_xy = self.world.sample_free_point(
                rng,
                self.world.target_radius,
                avoid=robot_xy,
                min_distance=self.min_start_goal_distance,
            )
        else:
            target_xy = np.asarray(target_xy, dtype=np.float64)

        theta = float(options.get("theta", rng.uniform(-np.pi, np.pi)))
        self.robot = RobotState(float(robot_xy[0]), float(robot_xy[1]), theta)
        self.target = target_xy
        self._steps = 0
        self.path_length = 0.0
        self._prev_distance = self.world.distance_to_target(self.robot, self.target)
        self.last_field_angle = self.robot.theta
        info = self._info(success=False, collision=False)
        if self.render_mode == "human":
            self.render()
        return self._observe(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        field_angle = action_to_field_angle(action)
        self.last_field_angle = field_angle
        previous = self.robot
        self.robot = integrate_step(
            self.robot,
            field_angle=field_angle,
            dt=self.dt,
            v_max=self.v_max,
            k_align=self.k_align,
            max_omega=self.max_omega,
            b_magnitude=self.b_magnitude,
        )
        self.path_length += float(
            np.hypot(self.robot.x - previous.x, self.robot.y - previous.y)
        )
        self._steps += 1

        distance = self.world.distance_to_target(self.robot, self.target)
        collision = self.world.collides(self.robot)
        success = (not collision) and self.world.reached_goal(self.robot, self.target)
        truncated = self._steps >= self.max_steps and not success and not collision
        terminated = success or collision

        reward = self.distance_scale * (self._prev_distance - distance)
        reward -= self.step_penalty
        if success:
            reward += self.goal_bonus
        if collision:
            reward += self.collision_penalty
        self._prev_distance = distance

        info = self._info(success=success, collision=collision)
        if self.render_mode == "human":
            self.render()
        return self._observe(), float(reward), terminated, truncated, info

    def _info(self, success: bool, collision: bool) -> dict[str, Any]:
        self.last_info = {
            "success": success,
            "collision": collision,
            "distance": self.world.distance_to_target(self.robot, self.target),
            "path_length": self.path_length,
            "steps": self._steps,
            "time": self._steps * self.dt,
            "field_angle": self.last_field_angle,
            "robot": (self.robot.x, self.robot.y, self.robot.theta),
            "target": (float(self.target[0]), float(self.target[1])),
        }
        return self.last_info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from simulation.renderer import MatplotlibRenderer

            self._renderer = MatplotlibRenderer(self.world_size, self.render_mode)
        return self._renderer.draw(
            robot=self.robot,
            target=self.target,
            obstacles=self.world.obstacles,
            robot_radius=self.robot_radius,
            target_radius=self.world.target_radius,
            field_angle=self.last_field_angle,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
