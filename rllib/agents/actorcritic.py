import numpy as np
import torch

from .agent import Agent
from rllib.basis import Basis
from rllib.policies import Policy

class ActorCritic(Agent):
    def __init__(self, vbasis:Basis, policy:Policy, alpha=0.001, valpha=0.01, lambda_=0.7, gamma=1.0, use_parl2=True):
        self.vbasis = vbasis
        self.policy = policy
        self.alpha = alpha
        self.valpha = valpha
        self.lambda_ = lambda_
        self.gamma = gamma
        self.use_parl2 = use_parl2

        if use_parl2:
            self.valpha = 1.0

        self.num_theta = self.policy.get_num_params()  # get_flat_params(self.actor).shape[0]
        self.n_featuresV = self.vbasis.getNumFeatures() # number of features for value funciton to use
        self.v = np.zeros(self.n_featuresV, dtype=np.float64).flatten()  # value function parameter vector
        self.ev = np.zeros_like(self.v)  # eligibility trace vector for value function
        self.etheta = np.zeros_like(self.num_theta, dtype=np.float64)  # eligibility trace vector for advantage function


    def get_action(self, obs, stochastic=True):
        '''
        sample action from policy in given state
        :param state: state to sample from
        :param stochastic: if false it uses the MLE action
        :return: tuple of (log probability of action, action)
        '''
        x = np.array(obs)
        a, logp = self.policy.get_action(x, stochastic)
        return a, logp

    def update(self, obs, act, blogp, reward, obs_next, terminal):
        '''
        update algorithm for both value function and policy
        :param blogp:
        :param reward: reward at time t
        :param terminal: bool for end of episode
        :return: tuple (if policy updated, advantage error)
        '''

        # get features for value function
        xt = self.vbasis.encode(np.array(obs))

        if not terminal:
            # get features for next state value function
            xtp1 = self.vbasis.encode(np.array(obs_next))
            xtp1 = xtp1.flatten().astype(np.float64)
            # value function of next state
            vtp1 = self.v.dot(xtp1)
        else:
            xtp1 = np.zeros_like((xt))
            # no next state so 0 for all
            # xtp1 = np.zeros_like(xt)
            vtp1 = 0
        # basis for policy
        xp = np.array(obs)  # self.pbasis.basify(np.array(state))
        at = np.array([act]).reshape(1, -1)
        # get compatible features
        grad_log_p, _ = self.policy.grad_logp(xp, at)

        # flatten everything (this was important for LSTD because it used c++)
        xt = xt.flatten().astype(np.float64)
        grad_log_p = grad_log_p.astype(np.float64).flatten()

        # value function of current state
        vt = self.v.dot(xt)

        # TD error
        delta = reward + self.gamma * vtp1 - vt

        # Decay eligibility traces
        self.ev = self.gamma * self.lambda_ * self.ev + xt
        self.etheta = self.gamma * self.lambda_ * self.etheta + grad_log_p

        # adaptive learning rate for valpha
        if self.use_parl2:
            d = self.ev.dot(xt)
            if d < 0:
                self.valpha = min(self.valpha, -1. / d)

        self.v += self.valpha * delta * self.ev

        normv = np.linalg.norm(self.v)
        if normv > 1000:
            self.v = self.v / normv

        gtheta = self.alpha * delta * self.etheta

        updated = False
        if not np.any(np.isnan(gtheta)):
            updated = True
            self.policy.add_to_params(gtheta)


        err = [delta]

        if terminal:
            # decay eligibility traces
            self.ev *= 0
            self.etheta *= 0

        return updated, err

    def new_episode(self):
        self.ev *= 0
        self.etheta *= 0
