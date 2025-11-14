import gymnasium as gym
from gymnasium import spaces

class DiscreteEnv(gym.Env):

    def __post_init__(self):
        assert self.action_space is not None, "Action space is undefined are you sure you setup the environment correctly?"
        assert isinstance(self.action_space, spaces.Discrete), "Action space should be discrete for DiscreteEnv"
