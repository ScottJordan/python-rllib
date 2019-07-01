import numpy as np
import torch


class Critic(object):
    def predict(self, xt):
        raise NotImplementedError

    def grad(self, xt):
        raise NotImplementedError

    def set_params(self, theta):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def pred_fn(self, theta, xt):
        raise NotImplementedError

    def grad_fn(self, theta, xt):
        raise NotImplementedError

    def get_num_params(self):
        raise NotImplementedError


class LinearVF(Critic):
    def __init__(self, basis):
        self.basis = basis
        self.num_features = basis.getNumBasisFunctions()
        self.theta = np.zeros(self.num_features, dtype=np.float64)

    def pred_fn(self, theta, xt):
        x = self.basis.basify(xt).flatten()
        return theta.dot(x)

    def grad_fn(self, theta, xt):
        return self.basis.basify(xt).flatten()

    def predict(self, xt):
        return self.pred_fn(self.theta, xt)

    def grad(self, xt):
        return self.grad_fn(self.theta, xt)

    def set_params(self, theta):
        self.theta = theta.flatten()

    def get_params(self):
        return self.theta

    def get_num_params(self):
        return self.theta.size

class LinearQ(Critic):
    def __init__(self, basis, num_actions):
        self.basis = basis
        self.num_features = basis.getNumBasisFunctions()
        self.num_actions = num_actions
        self.theta = np.zeros((self.num_features, num_actions), dtype=np.float64)

    def pred_fn(self, theta, xt):
        s, a = xt[0], xt[1]
        x = self.basis.basify(s).flatten()
        return theta[:, a].dot(x)

    def grad_fn(self, theta, xt):
        s, a = xt[0], xt[1]
        g = np.zeros_like(self.theta)
        g[:, a] = self.basis.basify(xt).flatten()
        return g

    def predict(self, xt):
        return self.pred_fn(self.theta, xt)

    def grad(self, xt):
        return self.grad_fn(self.theta, xt)

    def set_params(self, theta):
        self.theta = theta.reshape(self.num_features, self.num_actions)

    def get_params(self):
        return self.theta.reshape(-1)

    def get_num_params(self):
        return self.theta.size

class CompatibleFA(Critic):
    def __init__(self, policy):
        self.policy = policy
        self.num_features = policy.get_num_params()
        self.theta = np.zeros(self.num_features, dtype=np.float64)

    def pred_fn(self, theta, xt):
        s,a = xt[0], xt[1]
        _, gradlogp = self.policy.grad_logp(s, a.reshape(1, -1))
        return theta.dot(gradlogp.flatten().astype(np.float64))

    def grad_fn(self, theta, xt):
        _, g = self.policy.grad_logp(xt[0], xt[1].reshape(1, -1))
        return g.flatten().astype(np.float64)

    def predict(self, xt):
        return self.pred_fn(self.theta, xt)

    def grad(self, xt):
        return self.grad_fn(self.theta, xt)

    def set_params(self, theta):
        self.theta = theta.flatten()

    def get_params(self):
        return self.theta

    def get_num_params(self):
        return self.theta.size
