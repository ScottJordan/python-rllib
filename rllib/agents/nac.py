import numpy as np

from rllib.utils.utils import *
from .agent import Agent
from rllib.basis import Basis

class NACTD(Agent):
    def __init__(self, vbasis:Basis, policy,
                 eta=0.01, lam=0.7, gamma=1.0, beta=0.001):
        '''
        Natural Actor Critic TD algorithm
        :param vbasis: basis for value function
        :param policy: Policy class for the agent to use
        :param alpha: policy learning rate
        :param lam: eligibility decay rate
        :param gamma: reward discount factor
        '''

        self.alpha = 1 #critic learning rate
        self.eta = eta
        self.lam = lam # elgibility trace parameter
        self.gamma = gamma # reward discount factor
        self.beta = beta
        self.vbasis = vbasis
        self.policy = policy
        self.count = 0
        self.n_featuresV = self.vbasis.getNumFeatures()  # number of features for value funciton to use

        num_theta = self.policy.get_num_params()  # get_flat_params(self.policy).shape[0]
        self.v = np.zeros(self.n_featuresV, dtype=np.float64).flatten()  # value function parameter vector
        self.w = np.zeros(num_theta, dtype=np.float64).flatten() # advantage function parameter vector
        self.ev = np.zeros_like(self.v)  # eligibility trace vector for value function
        self.ew = np.zeros_like(self.w)  # eligibility trace vector for advantage function
        self.diagv = np.ones_like(self.v)
        self.diagw = np.ones_like(self.w)
        self.maxwnorm = 0.0


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

        self.count += 1
        # get features for value function

        xt = self.vbasis.encode(np.array(obs))
        if not terminal:
            # get features for next state value function
            xtp1 = self.vbasis.encode(np.array(obs_next))
            xtp1 = xtp1.flatten().astype(np.float64)
            # value function of next state
            vtp1 = self.v.dot(xtp1)
        else:
            # no next state so 0 for all
            xtp1 = np.zeros_like(xt)
            vtp1 = 0
        # basis for policy
        xp = np.array(obs)#self.pbasis.basify(np.array(state))
        at = np.array([act]).reshape(1, -1)
        # get compatible features
        grad_log_p, _ = self.policy.grad_logp(xp, at)

        xt = xt.flatten().astype(np.float64)
        grad_log_p = grad_log_p.astype(np.float64).flatten()

        # value function of current state
        vt = self.v.dot(xt)

        # advantage of current state action
        advt = self.w.dot(grad_log_p)
        # TD error
        delta = reward + self.gamma * vtp1 - advt -  vt

        # Decay eligibility traces
        self.ev = self.gamma * self.lam * self.ev + xt
        self.ew = self.gamma * self.lam * self.ew + grad_log_p

        dphi = self.gamma * xtp1 - xt
        beta = self.beta
        self.diagv += beta * (np.square(dphi) - self.diagv)
        self.diagw += beta * (np.square(grad_log_p) - self.diagw)


        d = self.ev.flatten().dot(dphi / np.sqrt(self.diagv)) + self.ew.dot(-grad_log_p / np.sqrt(self.diagw))
        if d < 0:
            self.alpha = min(self.alpha, -1. / d)
        
        self.w += 0.1 * self.alpha * delta * self.ew / np.sqrt(self.diagw)
        self.v += 0.1 * self.alpha * delta * self.ev / np.sqrt(self.diagv)


        err = [delta,]
        
        w = self.w # w is the natural policy gradient
        # udate policy every k steps
        if terminal:
            wnorm = np.linalg.norm(w)
            if wnorm > self.maxwnorm:
                print(wnorm)

            self.maxwnorm = max(wnorm, self.maxwnorm)
            eta = self.eta / (self.maxwnorm + 1e-8)
            dTheta = eta * w
            if not np.any(np.isnan(dTheta)):
                self.policy.add_to_params(dTheta)
                self.w -= dTheta

        if terminal:
            # decay eligibility traces
            self.ev *= 0
            self.ew *= 0

        return err

    def new_episode(self):
        # decay eligibility traces
        self.ev *= 0
        self.ew *= 0