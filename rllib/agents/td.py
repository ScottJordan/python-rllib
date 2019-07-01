
import numpy as np
from rllib.utils.utils import *
from .agent import Agent
from rllib.critics import Critic
from rllib.policies import Policy

from rllib.traces import TraceOptimizer

class TD(Agent):
    def __init__(self, policy:Policy, vf:Critic, topt:TraceOptimizer, gamma:float):
        self.policy = policy
        self.vf = vf
        self.topt = topt
        self.gamma = gamma

    def get_action(self, obs, stochastic=True):
        '''
        sample action from policy in given state
        :param state: state to sample from
        :param stochastic: if false it uses the MLE action
        :return: tuple of (log probability of action, action)
        '''
        x = np.array(obs)
        lgp, a = self.policy.get_action(x, stochastic)
        return lgp, a

    def update(self, obs, act, blogp, reward, obs_next, terminal):
        '''
        update algorithm for value function
        :param blogp:
        :param reward: reward at time t
        :param terminal: bool for end of episode
        :return: tuple (if policy updated, advantage error)
        '''
        xt = np.array(obs).flatten()

        if not terminal:
            # get features for next state value function
            xtp1 = np.array(obs_next).flatten()

            # value function of next state
            vtp1 = self.vf.predict(xtp1)
        else:
            xtp1 = np.zeros_like(xt)
            # no next state so 0 for all
            # xtp1 = np.zeros_like(xt)
            vtp1 = 0


        # value function of current state
        curV = self.vf.predict(xt)

        # TD error
        delta = reward + self.gamma * vtp1 - curV

        self.topt.update(delta, xt, xtp1, 0)
        self.vf.set_params(self.topt.theta)
        err = delta

        return err

    def new_episode(self):
        # decay eligibility traces
        self.topt.new_episode()


class TOTD(Agent):
    def __init__(self, policy:Policy, vf:Critic, topt:TraceOptimizer, gamma:float):
        self.policy = policy
        self.vf = vf
        self.topt = topt
        self.gamma = gamma
        self.first_step = True
        self.last_v = 0.

    def get_action(self, obs, stochastic=True):
        '''
        sample action from policy in given state
        :param state: state to sample from
        :param stochastic: if false it uses the MLE action
        :return: tuple of (log probability of action, action)
        '''
        x = np.array(obs)
        lgp, a = self.policy.get_action(x, stochastic)
        return lgp, a

    def update(self, obs, act, blogp, reward, obs_next, terminal):
        '''
        update algorithm for value function
        :param blogp:
        :param reward: reward at time t
        :param terminal: bool for end of episode
        :return: tuple (if policy updated, advantage error)
        '''
        xt = np.array(obs).flatten()

        if not terminal:
            # get features for next state value function
            xtp1 = np.array(obs_next).flatten()

            # value function of next state
            vtp1 = self.vf.predict(xtp1)
        else:
            xtp1 = np.zeros_like(xt)
            # no next state so 0 for all
            # xtp1 = np.zeros_like(xt)
            vtp1 = 0

        if self.first_step:
            # value function of current state
            self.last_v = self.vf.predict(xt)
            self.first_step = False

        vold = self.last_v
        # TD error
        delta = reward + self.gamma * vtp1 - vold

        self.topt.update(delta, xt, xtp1, vold)
        self.vf.set_params(self.topt.theta)
        self.last_v = vtp1
        err = delta

        return err

    def new_episode(self):
        # decay eligibility traces
        self.topt.new_episode()
        self.last_v = 0
        self.first_step = True

class TDQ(Agent):
    def __init__(self, policy:Policy, vf:Critic, advf:Critic, vopt:TraceOptimizer, aopt:TraceOptimizer, gamma:float, useAdvCv=True):
        self.policy = policy
        self.vf = vf
        self.advf = advf
        self.vopt = vopt
        self.aopt = aopt
        self.gamma = gamma
        self.useAdvCv = useAdvCv

    def get_action(self, obs, stochastic=True):
        '''
        sample action from policy in given state
        :param state: state to sample from
        :param stochastic: if false it uses the MLE action
        :return: tuple of (log probability of action, action)
        '''
        x = np.array(obs)
        lgp, a = self.policy.get_action(x, stochastic)
        return lgp, a

    def update(self, obs, act, blogp, reward, obs_next, terminal):
        '''
        update algorithm for value function
        :param blogp:
        :param reward: reward at time t
        :param terminal: bool for end of episode
        :return: tuple (if policy updated, advantage error)
        '''
        xt = np.array(obs).flatten()

        if not terminal:
            # get features for next state value function
            xtp1 = np.array(obs_next).flatten()

            # value function of next state
            vtp1 = self.vf.predict(xtp1)
        else:
            xtp1 = np.zeros_like(xt)
            # no next state so 0 for all
            # xtp1 = np.zeros_like(xt)
            vtp1 = 0

        # at = np.array([act]).reshape(1, -1)
        # get compatible features
        # _, grad_log_p = self.policy.grad_logp(xt, at).astype(np.float64).flatten()
        a = np.array([act]).reshape(1, -1)
        curV = self.vf.predict(xt)
        # TD error
        delta = reward + self.gamma * vtp1 - curV
        adv = self.advf.predict((xt, a))
        delta_advt = delta - adv

        if self.useAdvCv:
            delta = delta_advt
        self.old_w = np.copy(self.aopt.theta)
        self.vopt.update(delta, xt, xtp1, 0)
        self.vf.set_params(self.vopt.theta)
        self.aopt.update(delta, (xt, a), None, 0)
        self.advf.set_params(self.aopt.theta)

        err = delta

        return err

    def new_episode(self):
        # decay eligibility traces
        self.vopt.new_episode()
        self.aopt.new_episode()
        self.last_v = 0
        self.first_step = True
