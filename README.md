# AI-Based Autonomous Magnetic Microrobot for Targeted Drug Delivery

Autonomous 2D navigation of a magnetic microrobot using **reinforcement learning (PPO)** in simulation. The long-term pipeline is:

```
Microscope / camera → computer vision (YOLO / OpenCV) → robot pose
        → PPO agent → magnetic command → actuator → microrobot → target
```

This repository implements the first milestone: a **Gymnasium** environment, simplified magnetic physics, **Stable-Baselines3 PPO**, baselines (random / greedy heading), and evaluation metrics. Real hardware, YOLO, and continuous magnetic commands are staged for later versions.

## Goal

Classical magnetic microrobots are steered by a human operator. Here part of that loop is learned: given the robot pose and a target, the agent chooses a planar magnetic-field heading so the robot reaches the target.

**First milestone:** a PPO-controlled virtual microrobot reaches random start/goal poses in a 2D workspace. Experimental target: **≥ 90% success over 100 evaluation episodes** (not a guaranteed result of the first training run).

## Architecture (v0.1 / v0.2)

```
Simulation world (robot, target, optional obstacles)
        → Gymnasium env (state, reward, done)
        → PPO (PyTorch / Stable-Baselines3)
        → discrete magnetic heading (8 directions)
        → simplified 2D magnetic physics
        → updated robot pose
```

**State (normalized):** `[x, y, θ/π, Δx, Δy]`

**Actions (discrete):** field angle `φ ∈ {0°, 45°, …, 315°}`

**Reward:** distance progress + goal bonus + collision penalty + small step cost.

## Repository layout

```
├── config/config.yaml          # physics, reward, PPO hyperparameters
├── environment/                # Gymnasium env + magnetic physics
├── simulation/                 # world, obstacles, Matplotlib renderer
├── agent/                      # SB3 PPO wrapper
├── experiments/                # baselines and numbered studies
├── tests/                      # physics and env unit tests
├── train.py / evaluate.py / demo.py
├── models/checkpoints/         # trained policies (gitignored)
└── results/                    # logs, figures, tables
```

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest tests -q
```

## Train PPO

```bash
python train.py --timesteps 80000
```

Trains in an empty workspace by default. Use `--obstacles` for random rectangles/circles. The policy is saved to `models/checkpoints/ppo_microrobot_final.zip`.

## Evaluate

```bash
python evaluate.py --policy random --episodes 50 --csv results/tables/random.csv
python evaluate.py --policy heading --episodes 50 --csv results/tables/heading.csv
python evaluate.py --policy ppo --model models/checkpoints/ppo_microrobot_final.zip --episodes 100
```

**Metrics:** success rate, navigation time, path length, path efficiency, collision rate, final position error, control effort (timesteps used).

## Visualize one episode

```bash
python demo.py --policy heading --output results/figures/demo_heading.png
python demo.py --policy ppo --model models/checkpoints/ppo_microrobot_final.zip
```

## Experiments

| Script | Description |
|--------|-------------|
| `experiments/baseline.py` | Random vs greedy field-to-target heading |
| `experiments/experiment_01.py` | Train/eval PPO in empty workspace |
| `experiments/experiment_02.py` | Train/eval PPO with obstacles |
| `experiments/experiment_03.py` | Generalization over random starts (needs a trained zip) |

## Physics (first prototype)

Translation follows the applied field (gradient-pulling approximation). Heading aligns toward **B** with a clipped first-order torque model `τ ~ m × B`. Continuous `(φ, |B|)` actions and higher-fidelity hydrodynamics are planned for later versions.

## Roadmap

| Version | Feature |
|---------|---------|
| v0.1 | 2D robot + target |
| v0.2 | PPO navigation |
| v0.3 | Obstacles |
| v0.4 | Randomized environments |
| v0.5 | Richer magnetic-field model |
| v0.6 | Continuous actions |
| v0.7–v0.8 | PID baseline and PPO vs PID |
| v0.9–v1.0 | OpenCV / YOLO pose, image → control |
| v1.1–v1.2 | Higher-fidelity physics and experimental data |

## License

Research prototype. See the GitHub repository for collaboration details.
