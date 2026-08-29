"""Matplotlib renderer for the 2D microrobot workspace."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Rectangle

from environment.physics import RobotState
from simulation.world import Obstacle


class MatplotlibRenderer:
    def __init__(self, world_size: float, mode: str = "human") -> None:
        self.world_size = world_size
        self.mode = mode
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        if mode == "human":
            plt.ion()
            self.fig.show()

    def draw(
        self,
        robot: RobotState,
        target: np.ndarray,
        obstacles: list[Obstacle],
        robot_radius: float,
        target_radius: float,
        field_angle: float,
    ) -> np.ndarray | None:
        self.ax.clear()
        self.ax.set_xlim(0.0, self.world_size)
        self.ax.set_ylim(0.0, self.world_size)
        self.ax.set_aspect("equal")
        self.ax.set_title("Magnetic microrobot navigation")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        for obstacle in obstacles:
            if obstacle.kind == "circle":
                patch = Circle((obstacle.x, obstacle.y), obstacle.radius, color="#4b5563", alpha=0.85)
            else:
                patch = Rectangle(
                    (obstacle.x, obstacle.y),
                    obstacle.width,
                    obstacle.height,
                    color="#4b5563",
                    alpha=0.85,
                )
            self.ax.add_patch(patch)

        self.ax.add_patch(Circle((float(target[0]), float(target[1])), target_radius, color="#f59e0b", alpha=0.9))
        self.ax.add_patch(Circle((robot.x, robot.y), robot_radius, color="#2563eb", alpha=0.95))
        heading = FancyArrow(
            robot.x,
            robot.y,
            0.05 * math.cos(robot.theta),
            0.05 * math.sin(robot.theta),
            width=0.006,
            color="#1d4ed8",
        )
        field = FancyArrow(
            robot.x,
            robot.y,
            0.08 * math.cos(field_angle),
            0.08 * math.sin(field_angle),
            width=0.004,
            color="#dc2626",
        )
        self.ax.add_patch(heading)
        self.ax.add_patch(field)
        self.ax.legend(
            handles=[
                plt.Line2D([0], [0], color="#2563eb", lw=6, label="Robot"),
                plt.Line2D([0], [0], color="#f59e0b", lw=6, label="Target"),
                plt.Line2D([0], [0], color="#dc2626", lw=2, label="Magnetic field B"),
            ],
            loc="upper right",
        )
        self.fig.canvas.draw()
        if self.mode == "human":
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            return None
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def close(self) -> None:
        plt.close(self.fig)
