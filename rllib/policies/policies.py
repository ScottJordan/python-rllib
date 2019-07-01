import numpy as np
from typing import Tuple, Union
import gym


class Policy(object):
    def get_action(self, obs, stochastic:bool=True) -> Tuple[Union[int, float, np.ndarray, dict], float]:
        raise NotImplementedError

    def grad_logp(self, obs, action) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def add_to_params(self, grad: np.ndarray):
        raise NotImplementedError

    def get_params(self)-> np.ndarray:
        '''
        Gets the policy parameters
        :return: numpy vector
        '''
        raise NotImplementedError

    def set_params(self, params: np.ndarray):
        '''
        sets the policy parameters
        :param params: numpy vector for policy
        :return: none
        '''
        raise NotImplementedError

    def get_num_params(self)-> int:
        raise NotImplementedError

class RandomPolicy(Policy):
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def get_action(self, obs, stochastic: bool = True) -> Tuple[Union[int, float, np.ndarray, dict], float]:
        return self.action_space.sample(), 0.

    def grad_logp(self, obs, action) -> Tuple[np.ndarray, float]:
        return np.array([]), 0.

    def add_to_params(self, grad: np.ndarray):
        pass

    def get_params(self) -> np.ndarray:
        '''
        Gets the policy parameters
        :return: numpy vector
        '''
        return np.array([])

    def set_params(self, params: np.ndarray):
        '''
        sets the policy parameters
        :param params: numpy vector for policy
        :return: none
        '''
        pass

    def get_num_params(self) -> int:
        return 0
