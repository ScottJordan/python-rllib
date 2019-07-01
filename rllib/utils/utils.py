# -*- coding: utf-8 -*-

import numpy as np

import cloudpickle
import scipy.signal

def logRand(min, max, size=1):
    return np.exp(np.random.uniform(np.log(min), np.log(max), size=size))

def compute_cumulative_returns(rewards, Rtp1, discount):
    # This method builds up the cumulative sum of discounted rewards for each time step:
    # R[t] = sum_{t'>=t} γ^(t'-t)*r_t'
    # Note that we use γ^(t'-t) instead of γ^t'. This gives us a biased gradient but lower variance
    returns = []
    # Use the last baseline prediction to back up
    cum_return = Rtp1
    for reward in rewards[::-1]:
        cum_return = cum_return * discount + reward
        returns.append(cum_return)
    return returns[::-1]

def compute_advantages(rewards, baselines, discount, gae_lambda=1):
    # Given returns R_t and baselines b(s_t), compute (generalized) advantage estimate A_t
    deltas = rewards + discount * baselines[1:] - baselines[:-1]
    advs = []
    cum_adv = 0
    multiplier = discount * gae_lambda
    for delta in deltas[::-1]:
        cum_adv = cum_adv * multiplier + delta
        advs.append(cum_adv)
    return advs[::-1]


def compute_baselines(all_returns):
    """
    :param all_returns: A list of size T, where the t-th entry is a list of numbers, denoting the returns
    collected at time step t across different episodes
    :return: A vector of size T
    """
    baselines = np.zeros(len(all_returns))
    for t in range(len(all_returns)):
        "*** YOUR CODE HERE ***"
        if len(all_returns[t]) > 0:
            baselines[t] = np.mean(all_returns[t])
    return baselines

def explained_variance_1d(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 1e-8:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

