import gymnasium as gym
from gymnasium import spaces

class ContinuousEnv(gym.Env):

    def __post_init__(self):
        assert self.action_space is not None, "Action space is undefined are you sure you setup the environment correctly?"
        assert isinstance(self.action_space, spaces.Box), "Action space should be continuous for ContinuousEnv"