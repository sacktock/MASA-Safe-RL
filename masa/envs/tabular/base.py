import gymnasium as gym
from gymnasium import spaces

class TabularEnv(gym.Env):

    def __init__(self):
        self._transition_matrix = None
        self._successor_state_map = None
        self._prob_map = None

    def __post_init__(self):
        assert self.observation_space is not None, "Observation space is undefined are you sure you setup the environment correctly?"
        assert self.action_space is not None, "Action space is undefined are you sure you setup the environment correctly?"

        assert isinstance(self.observation_space, spaces.Discrete), "Observation space should be discrete for TabularEnv"
        assert isinstance(self.action_space, spaces.Discrete), "Action space should be discrete for TabularEnv"

    @property
    def has_transition_matrix(self):
        return self._transition_matrix is not None

    @property
    def has_successor_states_dict(self):
        return (self._successor_state_map is not None) and (self._prob_map is not None)

    def get_transition_matrix(self):
        if self.has_transition_matrix
            return self._transition_matrix
        else:
            return None

    def get_successor_states_dict(self):
        if self.has_successor_states_dict:
            return self._successor_state_map, self._prob_map
        else:
            return None
