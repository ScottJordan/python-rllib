from typing import Union, Dict, Tuple

import numpy as np
from collections import OrderedDict
from .agent import Agent
from rllib.policies import Policy

class RandomPolicySearch(Agent):
    def __init__(self, policy:Policy, sample_range:Tuple[float, float]=(0.1, 0.1), num_episodes:int=1, gamma:float=1.0, max_eps:int=1000):
        self.policy = policy
        self.sample_range = sample_range
        self.num_episodes = num_episodes
        self.max_eps = max_eps
        self.num_params = policy.get_num_params()
        self.gamma = gamma
        self.best_theta = None
        self.best_perf = -np.inf
        self.eps_sofar = 0
        params = self.sample_params()
        self.policy.set_params(params)
        self.Gs = []
        self.G = 0
        self.cur_gamma = 1.

    def get_action(self, obs: np.ndarray, stochastic: bool = True) -> Tuple[Union[int, np.ndarray, Dict], float]:
        a, logp = self.policy.get_action(obs, stochastic)
        return a, logp

    def update(self, obs: np.ndarray, act: Union[int, np.ndarray, Dict], blogp: float, reward: float, obs_next: np.ndarray, terminal: bool) -> Union[np.ndarray, None]:
        self.G += self.cur_gamma * reward
        self.cur_gamma *= self.gamma
        if terminal:
            self.Gs.append(self.G)
            self.G = 0
            self.eps_sofar += 1

        return None

    def new_episode(self):
        self.G = 0
        self.cur_gamma = 1.0

        if len(self.Gs) >= self.num_episodes:
            sample_new = True
            nG = np.mean(self.Gs)
            self.Gs = []
            if nG >= self.best_perf:
                self.best_theta = self.policy.get_params()
                self.best_perf = nG
        else:
            sample_new = False

        if sample_new:
            params = self.sample_params()
            self.policy.set_params(params)
            self.Gs = []

        if self.eps_sofar >= self.max_eps:
            self.policy.set_params(self.best_theta)


    def sample_params(self):
        return np.random.uniform(self.sample_range[0], self.sample_range[1], size=self.num_params)
