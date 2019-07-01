from typing import Union, Dict, Tuple

import numpy as np
from collections import OrderedDict
from .agent import Agent
from rllib.policies import Policy

class BasicRandomSearch(Agent):
    def __init__(self, policy:Policy, alpha:float, num_samples:int =20, std:float = 0.05, gamma:float = 1.0, max_eps:int = 1000):
        self.policy = policy
        self.num_samples = num_samples
        self.alpha = alpha
        self.std = std
        self.max_eps = max_eps
        self.num_params = policy.get_num_params()
        self.gamma = gamma

        self.theta = self.policy.get_params()
        self.eps_sofar = 0

        self.deltas = np.array(self.sample_deltas())
        self.dnum = 0
        self.pn = 0
        #self.policy.set_params(self.theta - self.deltas[self.dnum])
        self.Gs = np.zeros((self.num_samples, 2))
        self.G = 0
        self.cur_gamma = 1.

    def act(self, env, stochastic=True, train=True):
        obs = env.state
        act, blogp = self.policy.get_action(obs, stochastic)
        env.step(act)
        reward = env.reward
        terminal = env.done
        self.G += self.cur_gamma * reward
        self.cur_gamma *= self.gamma
        if terminal and train:
            self.Gs[self.dnum, self.pn] = self.G
            self.G = 0
            self.eps_sofar += 1
            if self.pn == 0:
                self.pn = 1
            else:
                self.pn = 0
                self.dnum += 1

    def get_action(self, obs: np.ndarray, stochastic: bool = True) -> Tuple[Union[int, np.ndarray, Dict], float]:
        a, logp = self.policy.get_action(obs, stochastic)
        return a, logp

    def update(self, obs: np.ndarray, act: Union[int, np.ndarray, Dict], blogp: float, reward: float, obs_next: np.ndarray, terminal: bool) -> Union[np.ndarray, None]:
        self.G += self.cur_gamma * reward
        self.cur_gamma *= self.gamma
        if terminal:
            self.Gs[self.dnum, self.pn] = self.G
            self.G = 0
            self.eps_sofar += 1
            if self.pn == 0:
                self.pn = 1
            else:
                self.pn = 0
                self.dnum += 1

        return None

    def new_episode(self):
        self.G = 0
        self.cur_gamma = 1.0

        if self.eps_sofar < self.max_eps:
            if self.dnum >= self.num_samples:
                self.compute_theta()
                self.deltas = np.array(self.sample_deltas())
                self.dnum = 0
                self.pn = 0

            if self.pn == 0:  # negative delta
                self.policy.set_params(self.theta - self.deltas[self.dnum])
            else:
                self.policy.set_params(self.theta + self.deltas[self.dnum])

        else:
            self.policy.set_params(self.theta)


    def sample_deltas(self):
        return [np.random.normal(loc=0, scale=self.std, size=self.num_params) for _ in range(self.num_samples)]

    def compute_theta(self):
        g = (self.Gs[:, 1] - self.Gs[:, 0]).reshape(-1, 1) * self.deltas.reshape(-1, self.num_params)
        ag = np.mean(g, axis=0)
        self.theta = self.theta + self.alpha * ag

class AugmentedRandomSearch(Agent):
    def __init__(self, policy:Policy, alpha:float, num_samples:int =20, std:float = 0.05, beta:int = 10, gamma:float = 1.0, max_eps:int = 1000):
        self.policy = policy
        self.num_samples = num_samples
        self.alpha = alpha
        self.beta = beta
        self.std = std
        self.max_eps = max_eps
        self.num_params = policy.get_num_params()
        self.gamma = gamma

        self.theta = self.policy.get_params()
        self.eps_sofar = 0

        self.deltas = np.array(self.sample_deltas())
        self.dnum = 0
        self.pn = 0
        self.policy.set_params(self.theta - self.deltas[self.dnum])
        self.Gs = np.zeros((self.num_samples, 2))
        self.G = 0
        self.cur_gamma = 1.

    def get_action(self, obs: np.ndarray, stochastic: bool = True) -> Tuple[Union[int, np.ndarray, Dict], float]:
        a, logp = self.policy.get_action(obs, stochastic)
        return a, logp

    def update(self, obs: np.ndarray, act: Union[int, np.ndarray, Dict], blogp: float, reward: float, obs_next: np.ndarray, terminal: bool) -> Union[np.ndarray, None]:
        self.G += self.cur_gamma * reward
        self.cur_gamma *= self.gamma
        if terminal:
            self.Gs[self.dnum, self.pn] = self.G
            self.G = 0
            self.eps_sofar += 1
            if self.pn == 0:
                self.pn = 1
            else:
                self.pn = 0
                self.dnum += 1

        return None

    def new_episode(self):
        self.G = 0
        self.cur_gamma = 1.0

        if self.eps_sofar < self.max_eps:
            if self.dnum >= self.num_samples:
                self.compute_theta()
                self.deltas = np.array(self.sample_deltas())
                self.dnum = 0
                self.pn = 0

            if self.pn == 0:  # negative delta
                self.policy.set_params(self.theta - self.deltas[self.dnum])
            else:
                self.policy.set_params(self.theta + self.deltas[self.dnum])

        else:
            self.policy.set_params(self.theta)


    def sample_deltas(self):
        return [np.random.normal(loc=0, scale=self.std, size=self.num_params) for _ in range(self.num_samples)]

    def compute_theta(self):
        idx = np.argsort(self.Gs.max(axis=1))[:self.beta]
        g = (self.Gs[idx, 1] - self.Gs[idx, 0]).reshape(-1, 1) * self.deltas[idx, :].reshape(-1, self.num_params)
        ag = np.mean(g, axis=0)
        sigma = np.std(self.Gs)
        if not np.isfinite(sigma):
            sigma = 1.0
        if sigma < 0.01:
            sigma = 1. # not in paper
        self.theta = self.theta + (self.alpha / sigma) * ag