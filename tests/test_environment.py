import numpy as np

from environment.microrobot_env import MicrorobotEnv
from environment.physics import RobotState
from simulation.world import Obstacle, SimulationWorld


def test_reset_and_observation_shape():
    env = MicrorobotEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,)
    assert env.observation_space.contains(obs)
    assert "distance" in info
    env.close()


def test_step_changes_state():
    env = MicrorobotEnv()
    env.reset(seed=1, options={"robot": [0.2, 0.2], "target": [0.8, 0.2], "theta": 0.0})
    before = (env.robot.x, env.robot.y)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (5,)
    assert isinstance(reward, float)
    assert env.robot.x != before[0] or env.robot.y != before[1]
    assert not (terminated and truncated)
    env.close()


def test_goal_bonus_when_close():
    env = MicrorobotEnv()
    env.reset(seed=2, options={"robot": [0.50, 0.50], "target": [0.52, 0.50], "theta": 0.0})
    env.robot = RobotState(0.50, 0.50, 0.0)
    env.target = np.array([0.52, 0.50])
    env._prev_distance = env.world.distance_to_target(env.robot, env.target)
    _, reward, terminated, _, info = env.step(0)
    assert info["success"] or reward > 0
    env.close()


def test_rect_obstacle_collision():
    world = SimulationWorld()
    world.obstacles = [Obstacle(kind="rect", x=0.4, y=0.4, width=0.2, height=0.2)]
    assert world.hits_obstacle(0.45, 0.45, 0.02)
    assert not world.hits_obstacle(0.1, 0.1, 0.02)
