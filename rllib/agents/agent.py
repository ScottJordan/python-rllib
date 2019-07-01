import numpy as np

from typing import Union, Tuple, Dict

class Agent(object):
    def get_action(self, obs:np.ndarray, stochastic:bool=True)-> Tuple[Union[int, np.ndarray, Dict], float]:
        raise NotImplementedError

    def update(self, obs:np.ndarray, act:Union[int, np.ndarray, Dict], blogp:float, reward:float, obs_next:np.ndarray, terminal:bool)->Union[np.ndarray, None]:
        raise NotImplementedError

    def new_episode(self):
        raise NotImplementedError


class FixedAgent(Agent):
    def __init__(self, policy):
        self.policy = policy

    def get_action(self, obs:np.ndarray, stochastic:bool=True)-> Tuple[Union[int, np.ndarray, Dict], float]:
        return self.policy.get_action(obs, stochastic)

    def update(self, obs:np.ndarray, act:Union[int, np.ndarray, Dict], blogp:float, reward:float, obs_next:np.ndarray, terminal:bool)->Union[np.ndarray, None]:
        return None

    def new_episode(self):
        pass