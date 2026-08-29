"""PPO policy notes.

Stable-Baselines3 MlpPolicy already implements an actor-critic MLP.
The default architecture for this project is:

    state (5) -> Linear(128) -> ReLU -> Linear(128) -> ReLU
      -> policy head (8 discrete magnetic headings)
      -> value head (scalar)

A custom PyTorch policy can replace SB3 later without changing the env API.
"""

DEFAULT_NET_ARCH = [128, 128]
