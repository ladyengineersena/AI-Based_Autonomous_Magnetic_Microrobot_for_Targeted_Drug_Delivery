"""Simulation world: robot, target, obstacles, and collisions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment.physics import RobotState


@dataclass
class Obstacle:
    kind: str  # "rect" or "circle"
    x: float
    y: float
    width: float = 0.1
    height: float = 0.1
    radius: float = 0.05


@dataclass
class SimulationWorld:
    world_size: float = 1.0
    robot_radius: float = 0.025
    target_radius: float = 0.04
    goal_threshold: float = 0.05
    wall_collision: bool = True
    obstacles: list[Obstacle] = field(default_factory=list)

    def in_bounds(self, x: float, y: float, radius: float) -> bool:
        return (
            radius <= x <= self.world_size - radius
            and radius <= y <= self.world_size - radius
        )

    def hits_obstacle(self, x: float, y: float, radius: float) -> bool:
        for obstacle in self.obstacles:
            if obstacle.kind == "circle":
                dx = x - obstacle.x
                dy = y - obstacle.y
                if dx * dx + dy * dy <= (radius + obstacle.radius) ** 2:
                    return True
            else:
                closest_x = min(max(x, obstacle.x), obstacle.x + obstacle.width)
                closest_y = min(max(y, obstacle.y), obstacle.y + obstacle.height)
                dx = x - closest_x
                dy = y - closest_y
                if dx * dx + dy * dy <= radius * radius:
                    return True
        return False

    def collides(self, state: RobotState) -> bool:
        if self.wall_collision and not self.in_bounds(state.x, state.y, self.robot_radius):
            return True
        return self.hits_obstacle(state.x, state.y, self.robot_radius)

    def reached_goal(self, state: RobotState, target: np.ndarray) -> bool:
        dx = target[0] - state.x
        dy = target[1] - state.y
        return (dx * dx + dy * dy) ** 0.5 < self.goal_threshold

    def distance_to_target(self, state: RobotState, target: np.ndarray) -> float:
        dx = target[0] - state.x
        dy = target[1] - state.y
        return float((dx * dx + dy * dy) ** 0.5)

    def sample_free_point(
        self,
        rng: np.random.Generator,
        radius: float,
        avoid: np.ndarray | None = None,
        min_distance: float = 0.0,
        max_tries: int = 200,
    ) -> np.ndarray:
        for _ in range(max_tries):
            point = rng.uniform(radius, self.world_size - radius, size=2)
            dummy = RobotState(float(point[0]), float(point[1]), 0.0)
            if self.collides(dummy):
                continue
            if avoid is not None:
                delta = point - avoid
                if float(np.linalg.norm(delta)) < min_distance:
                    continue
            return point.astype(np.float64)
        return np.array([self.world_size * 0.2, self.world_size * 0.2], dtype=np.float64)

    def spawn_obstacles(self, rng: np.random.Generator, n: int, min_size: float, max_size: float) -> None:
        self.obstacles = []
        for _ in range(n):
            kind = "rect" if rng.random() < 0.7 else "circle"
            size = float(rng.uniform(min_size, max_size))
            if kind == "circle":
                margin = size + self.robot_radius
                x = float(rng.uniform(margin, self.world_size - margin))
                y = float(rng.uniform(margin, self.world_size - margin))
                self.obstacles.append(Obstacle(kind="circle", x=x, y=y, radius=size))
            else:
                x = float(rng.uniform(0.05, self.world_size - size - 0.05))
                y = float(rng.uniform(0.05, self.world_size - size - 0.05))
                self.obstacles.append(
                    Obstacle(kind="rect", x=x, y=y, width=size, height=size * float(rng.uniform(0.6, 1.2)))
                )
