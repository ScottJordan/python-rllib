import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.autograd as autog

from rllib.utils.utils import *

from .agent import Agent

class CEM(Agent):
    def __init__(self, policy, population, numElite, sigma, gamma):
        '''
        Natural Actor Critic TD algorithm
        :param vbasis: basis for value function
        :param policy: Policy class for the agent to use
        :param alpha: actor learning rate
        :param calpha: critic learning rate
        :param lambda_: eligibility decay rate
        :param gamma: reward discount factor
        :param k: policy update rate
        '''
        self.actor = policy
        self.population = population
        self.numElite = numElite
        self.num_params = policy.get_num_params()
        self.sigma = sigma
        self.Sigma = sigma * np.eye(self.num_params, self.num_params)
        self.theta = policy.get_params()
        self.gamma = gamma
        self.samples = []
        self.samples_ret = []
        self.cur_gamma = 1.
        self.sample_count = 0
        self.terminal_state = False
        self.sample_params()

        if torch.cuda.is_available():
            x = torch.ones(30).cuda()
            x *= 10
            y = x.cpu()
            del x
            del y

    def sample_params(self):
        samples = np.random.multivariate_normal(self.theta, self.Sigma, size=self.population)
        self.samples = [np.copy(samples[i,:]) for i in range(self.population)]
        self.samples_ret = [0. for _ in range(self.population)]
        self.sample_count = 0
        self.cur_gamma = 1.
        self.actor.set_params(self.samples[0].flatten())

    def get_action(self, obs, stochastic=True):
        '''
        sample action from policy in given state
        :param state: state to sample from
        :param stochastic: if false it uses the MLE action
        :return: tuple of (log probability of action, action)
        '''
        x = np.array(obs)
        lgp, a = self.actor.get_action(x, stochastic)
        return lgp, a

    def update(self, obs, act, blogp, reward, obs_next, terminal):
        '''
        update algorithm for both value function and policy
        :param blogp:
        :param reward: reward at time t
        :param terminal: bool for end of episode
        :return: tuple (if policy updated, advantage error)
        '''

        self.samples_ret[self.sample_count] += self.cur_gamma * reward
        self.cur_gamma *= self.gamma

        if terminal:
            self.terminal_state = True

        return 0, [0., 0.]

    def new_episode(self):

        self.sample_count += 1

        if self.sample_count == self.population:
            self.compute_stats()
            self.sample_params()
        else:
            self.actor.set_params(self.samples[self.sample_count])
        self.terminal_state = False

    def compute_stats(self):
        idx = np.argsort(self.samples_ret)[::-1][:self.numElite]
        elite = np.array([self.samples[i] for i in idx])
        self.theta = np.mean(elite, axis=0)
        Sigma = self.sigma * np.eye(self.num_params, self.num_params)
        diff = elite - self.theta.reshape(1, -1)
        for i in range(self.numElite):
            Sigma += np.outer(diff[i, :], diff[i, :])
        Sigma /= float(self.numElite + self.sigma)
        self.Sigma = Sigma




