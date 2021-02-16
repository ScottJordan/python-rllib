import numpy as np
import torch

from .agent import Agent
from rllib.basis import Basis
from rllib.policies import Policy

class Sarsa(Agent):
    def __init__(self, basis:Basis, num_actions:int, epsilon=0.015, lambda_=0.8, gamma=1.0):
        self.basis = basis
        self.num_actions = num_actions
        self.epsilon=epsilon
        self.alpha = 1.0
        self.lambda_ = lambda_
        self.gamma = gamma

        self.n_features = self.basis.getNumFeatures() # number of features for value function to use
        self.q = np.zeros((self.n_features, num_actions), dtype=np.float64)  # parameters for q function
        self.eq = np.zeros_like(self.q)  # eligibility trace vector for q function

        self.last_action = None
        self.last_logp = None

    def epsilon_greedy(self, x, stochastic=True):
        qs = np.dot(x, self.q).flatten()
        amax = np.argmax(qs)
        if stochastic:
            r = np.random.rand()
            if r < self.epsilon:
                a = np.random.choice(range(self.num_actions))
            else:
                a = amax
            logp = self.epsilon / self.num_actions + (a == amax) * (1 - self.epsilon)
        else:
            a = amax
            logp = 0.0
        return a, logp, qs[a]


    def get_action(self, obs, stochastic=True):
        '''
        sample action from policy in given state
        :param state: state to sample from
        :param stochastic: if false it uses the MLE action
        :return: tuple of (log probability of action, action)
        '''
        if self.last_action is not None:
            return self.last_action, self.last_logp
        else:
            x = np.array(obs)
            x = self.basis.encode(x).flatten().astype(np.float64)
            return self.epsilon_greedy(x, stochastic)[:2]


    def update(self, obs, act, blogp, reward, obs_next, terminal):
        '''
        update algorithm step for q function and choose next action
        :param blogp:
        :param reward: reward at time t
        :param terminal: bool for end of episode
        :return: tuple (if policy updated, advantage error)
        '''

        xt = self.basis.encode(np.array(obs)).astype(np.float64).flatten()  # get features for value function
        qt = self.q[:, act].dot(xt)  # get current prediction of q value
        phit = np.zeros_like(self.q)
        phit[:, act] = xt
        phit = phit

        if not terminal:
            xtp1 = self.basis.encode(np.array(obs_next)).astype(np.float64).flatten()
            self.last_action, self.last_logp, qtp1 = self.epsilon_greedy(xtp1)  # select next action and get value
            phitp1 = np.zeros_like(self.q)
            phitp1[:, self.last_action] = xtp1
            phitp1 = phitp1
        else:
            qtp1 = 0
            phitp1 = np.zeros_like(phit)

        # TD error
        delta = reward + self.gamma * qtp1 - qt


        # Decay eligibility traces
        self.eq = self.gamma * self.lambda_ * self.eq + phit

        # adaptive learning rate for alpha
        d = self.eq.flatten().dot(self.gamma * phitp1.flatten() - phit.flatten())
        if d < 0:
            self.alpha = min(self.alpha, -1. / d)


        self.q += self.alpha * delta * self.eq

        err = [delta]

        if terminal:
            # decay eligibility traces
            self.eq *= 0
            self.last_action = None
            self.last_logp = None

        return True, err

    def new_episode(self):
        self.eq *= 0
        self.last_action = None
        self.last_logp = None
