import numpy as np
from rllib.critics import Critic

class TraceOptimizer(object):
    def update(self, delta, xt, xtp1, oldP):
        raise NotImplementedError

    def new_episode(self):
        raise NotImplementedError

class AccumulatingTraceOptimizer(TraceOptimizer):
    def __init__(self, vf:Critic, alpha:float, decay:float):
        self.vf = vf
        self.theta = vf.get_params()
        self.e = np.zeros_like(self.theta)
        self.alpha = alpha
        self.decay = decay

    def update(self, delta:float, xt, xtp1, oldP):
        grad = self.vf.grad_fn(self.theta, xt)
        self.e = self.decay * self.e + grad
        self.theta[:] += self.alpha * delta * self.e

    def new_episode(self):
        self.e *= 0

class VesTraceOptimizer(TraceOptimizer):
    def __init__(self, vf:Critic, gamma:float, lam: float, per_param=False):
        self.vf = vf
        self.theta = vf.get_params()
        self.e = np.zeros_like(self.theta)
        self.alpha = 1.0
        self.decay = gamma * lam
        self.gamma = gamma
        self.per_param = per_param
        if per_param:
            self.v = np.zeros_like(self.theta)
            self.h = np.ones_like(self.theta)
        else:
            self.v = 0.0#np.zeros_like(self.theta)
            self.h = 1.0  # np.ones_like(self.theta)
        self.g = np.ones_like(self.theta)

        self.tau = 2.0
        self.tau0 = 1.0

    def update(self, delta:float, xt, xtp1, oldP):
        grad = self.vf.grad_fn(self.theta, xt)
        # gp1 = self.vf.grad_fn(self.theta, xtp1)
        # dphi = self.gamma * gp1 - grad
        dphi = -grad
        self.e = self.decay * self.e + grad
        de = delta * self.e
        self.g = self.g + (1. / self.tau) * (de - self.g)
        if self.per_param:
            self.v = self.v + (1. / self.tau) * (np.square(de) - self.v)
            self.h = self.h + (1. / self.tau) * (np.square(dphi) - self.h)
            self.alpha = np.clip(np.sqrt(np.square(self.g)) / np.sqrt(self.v * self.h), a_min=None, a_max=1)
            self.tau0 = self.tau0 + np.square(self.alpha).mean()
            self.tau = (1. - (np.square(self.g).sum() / self.v.sum())) * self.tau + self.tau0
        else:
            self.v = self.v + (1. / self.tau) * (np.square(de).sum() - self.v)
            self.h = self.h + (1. / self.tau) * (np.square(dphi).sum() - self.h)
            self.alpha = np.clip(np.sqrt(np.square(self.g).sum()) / np.sqrt(self.v * self.h), a_min=None, a_max=1)
            self.tau0 = self.tau0 + 1#np.square(self.alpha)
            self.tau = (1. - (np.square(self.g).sum() / self.v)) * self.tau + self.tau0
        # self.alpha = np.clip(np.square(self.g).sum()/self.v.dot(self.h), a_min=0.0, a_max=1.0)

        self.tau = np.clip(self.tau, 2., 1e6)

        self.theta[:] += self.alpha * delta * self.e

    def new_episode(self):
        self.e *= 0

class TOVesTraceOptimizer(TraceOptimizer):
    def __init__(self, vf:Critic, gamma:float, lam: float, per_param=False, tau0_fn=lambda x: 1):
        self.vf = vf
        self.theta = vf.get_params()
        self.theta_old= np.copy(self.theta)
        self.e = np.zeros_like(self.theta)
        self.alpha = 1.0
        self.decay = gamma * lam
        self.gamma = gamma
        self.per_param = per_param
        # self.use_alpha = use_alpha
        self.tau0_fn = tau0_fn
        if per_param:
            self.v = np.ones_like(self.theta)
            self.h = np.ones_like(self.theta)
        else:
            self.v = 1.0#np.zeros_like(self.theta)
            self.h = 1.0  # np.ones_like(self.theta)
        self.g = np.zeros_like(self.theta)

        self.tau = 2.0
        self.tau0 = 1.0

    def update(self, delta:float, xt, xtp1, oldP):
        curV = self.vf.pred_fn(self.theta, xt)
        grad = self.vf.grad_fn(self.theta, xt)
        #gp1 = self.vf.grad_fn(self.theta, xtp1)
        # dphi = self.gamma * gp1 - grad
        dphi = grad
        dece = self.decay * self.e + grad - self.decay * self.e.dot(grad) * grad
        de = delta * (dece) - (curV - oldP) * grad
        # self.e = self.decay * self.e + grad?

        # de = delta * self.e
        t = self.tau
        self.g = self.g + (1. / t) * (de - self.g)
        if self.per_param:
            self.v = self.v + (1. / t) * (np.square(de) - self.v)
            self.h = self.h + (1. / t) * (np.square(dphi) - self.h)
            alpha = np.sqrt(np.square(self.g)) / np.sqrt(self.v * self.h)
            self.alpha = np.clip(alpha, a_min=None, a_max=1.0)
            self.tau0 += self.tau0_fn(self.alpha)#self.tau0 + np.square(self.alpha)
            self.tau = (1. - (np.square(self.g) / self.v)) * self.tau + self.tau0
        else:
            self.v = self.v + (1. / t) * (np.square(de).sum() - self.v)
            self.h = self.h + (1. / t) * (np.square(dphi).sum() - self.h)
            self.alpha = np.clip(np.sqrt(np.square(self.g).sum()) / np.sqrt(self.v * self.h), a_min=None, a_max=1.0)
            # self.alpha = np.clip(np.square(self.g).sum() / (self.v * self.h), a_min=None, a_max=1.0)
            # if self.use_alpha:
            #     self.tau0 = self.tau0 + np.sqrt(self.alpha)#np.square(self.alpha)
            # else:
            #     self.tau0 = self.tau0 + 1
            self.tau0 += self.tau0_fn(self.alpha)
            self.tau = (1. - (np.square(self.g).sum() / self.v)) * self.tau + self.tau0
        # self.g = self.g + (1. / self.tau) * (de - self.g)
        # if self.per_param:
        #     self.v = self.v + (1. / self.tau) * (np.square(de) - self.v)
        #     self.h = self.h + (1. / self.tau) * (np.square(dphi) - self.h)
        #     self.alpha = np.clip(np.sqrt(np.square(self.g)) / np.sqrt(self.v * self.h), a_min=None, a_max=1)
        #     self.tau0 = self.tau0 + np.square(self.alpha).mean()
        #     self.tau = (1. - (np.square(self.g).sum() / self.v.sum())) * self.tau + self.tau0
        # else:
        #     self.v = self.v + (1. / self.tau) * (np.square(de).sum() - self.v)
        #     self.h = self.h + (1. / self.tau) * (np.square(dphi).sum() - self.h)
        #     self.alpha = np.clip(np.sqrt(np.square(self.g).sum()) / np.sqrt(self.v * self.h), a_min=None, a_max=1)
        #     self.tau0 = self.tau0 + 1  # np.square(self.alpha)
        #     self.tau = (1. - (np.square(self.g).sum() / self.v)) * self.tau + self.tau0
        # self.alpha = np.clip(np.square(self.g).sum()/self.v.dot(self.h), a_min=0.0, a_max=1.0)

        self.tau = np.clip(self.tau, 2., 1e6)
        self.e = self.decay * self.e + grad - self.alpha * self.decay * self.e.dot(grad) * grad
        self.theta_old = np.copy(self.theta)
        self.theta[:] += self.alpha * delta * self.e - self.alpha * (curV - oldP) * grad

    def new_episode(self):
        self.e *= 0
        self.theta_old = np.copy(self.theta)

class DutchTraceOptimzier(TraceOptimizer):
    def __init__(self, vf, alpha, decay):
        self.vf = vf
        self.theta = vf.get_params()
        self.e = np.zeros_like(self.theta)
        self.alpha = alpha
        self.decay = decay

    def update(self, delta, xt, xtp1, oldP):
        grad = self.vf.grad_fn(self.theta, xt)
        self.e = self.decay * self.e + self.alpha * grad - self.alpha * self.decay * self.e.dot(grad) * grad
        self.theta[:] += self.e * delta

    def new_episode(self):
        self.e *= 0.

class TOTraceOptimizer(TraceOptimizer):
    def __init__(self, vf, alpha, decay):
        self.vf = vf
        self.theta = vf.get_params()
        self.theta_old = np.copy(self.theta)
        self.e = np.zeros_like(self.theta)
        self.alpha = alpha
        self.decay = decay

    def update(self, delta, xt, xtp1, oldP):
        curV = self.vf.pred_fn(self.theta, xt)
        grad = self.vf.grad_fn(self.theta, xt)
        # self.alpha = logRand(1e-4, 1.0, size=grad.shape)
        self.e = self.decay * self.e + self.alpha * grad - self.alpha * self.decay * self.e.dot(grad) * grad

        self.theta_old = np.copy(self.theta)
        self.theta[:] += self.e * delta - self.alpha * (curV - oldP) * grad

    def new_episode(self):
        self.e *= 0
        self.theta_old = np.copy(self.theta)
