"""Simplified 2D magnetic microrobot physics.

The first prototype uses a gradient-pulling approximation:
the applied field direction sets the translation direction, while
heading aligns toward B with a first-order torque model tau ~ m x B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

N_DISCRETE_ACTIONS = 8
DISCRETE_ANGLES = np.linspace(0.0, 2.0 * math.pi, N_DISCRETE_ACTIONS, endpoint=False)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def action_to_field_angle(action: int) -> float:
    """Map a discrete action index to a magnetic field heading in radians."""
    if action < 0 or action >= N_DISCRETE_ACTIONS:
        raise ValueError(f"Action {action} is outside [0, {N_DISCRETE_ACTIONS - 1}]")
    return float(DISCRETE_ANGLES[action])


def magnetic_field_vector(phi: float, magnitude: float = 1.0) -> np.ndarray:
    """Return B = |B| [cos phi, sin phi]."""
    return magnitude * np.array([math.cos(phi), math.sin(phi)], dtype=np.float64)


@dataclass
class RobotState:
    x: float
    y: float
    theta: float
    v: float = 0.0
    omega: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.omega], dtype=np.float64)


def integrate_step(
    state: RobotState,
    field_angle: float,
    dt: float,
    v_max: float,
    k_align: float,
    max_omega: float,
    b_magnitude: float = 1.0,
) -> RobotState:
    """Advance the robot one timestep under a uniform planar field.

    Translation follows the field (gradient-pulling approximation).
    Angular velocity is a clipped proportional alignment toward B:
        omega = clip(k_align * wrap(phi - theta), +/- max_omega)
    """
    heading_error = wrap_angle(field_angle - state.theta)
    omega = float(np.clip(k_align * heading_error, -max_omega, max_omega))
    theta = wrap_angle(state.theta + omega * dt)

    speed = v_max * float(np.clip(b_magnitude, 0.0, 1.0))
    vx = speed * math.cos(field_angle)
    vy = speed * math.sin(field_angle)

    return RobotState(
        x=state.x + vx * dt,
        y=state.y + vy * dt,
        theta=theta,
        v=speed,
        omega=omega,
    )
