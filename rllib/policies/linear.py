from .policies import Policy
from rllib.basis import Basis
from collections import OrderedDict
import numpy as np
from typing import Tuple, Union, Dict

class Linear_Softmax(Policy):
    def __init__(self, basis:Basis, n_actions:int):
        super(Linear_Softmax, self).__init__()
        self.basis = basis
        self.n_actions = n_actions
        self.n_inputs =  basis.getNumFeatures()

        self.basis = basis
        self.theta = np.zeros((self.n_inputs, self.n_actions))


        self.num_params = int(self.theta.size)


    def get_action(self, obs:np.ndarray, stochastic:bool=True)->Tuple[int, float]:
        x = self.basis.encode(obs)

        p = self.get_p(x)

        if stochastic:
            a = int(np.random.choice(range(p.shape[0]), p=p, size=1))
            logp = float(np.log(p[a]))
        else:
            a = int(np.argmax(p))
            logp = 0.

        return a, logp

    def log_probabilty(self, obs:np.ndarray, action: int)->float:
        x = self.basis.encode(obs)
        p = self.get_p(x)
        return np.log(p[action])

    def get_num_params(self):
        return self.num_params

    def get_params(self):
        return self.theta.flatten()

    def add_to_params(self, grad):
        self.theta += grad.reshape(self.theta.shape)

    def set_params(self, params):
        self.theta = params.reshape(self.theta.shape)

    def grad_logp(self, obs: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        x = self.basis.encode(obs)
        theta = self.theta
        a = np.zeros(self.n_actions, dtype=theta.dtype)
        a[action] = 1
        u = self.get_p(x, theta)
        gtheta = x.reshape(-1, 1) * (a - u).reshape(1, -1)  # |s|x1 * 1x|a| -> |s|x|a|
        logp = np.log(u[action])

        return gtheta.flatten(), logp

    def get_p(self, x, theta=None):
        if not isinstance(theta, np.ndarray):
            theta = self.theta
        u = np.exp(np.clip(np.dot(x, theta), -32, 32))
        u /= u.sum()

        return u

class Linear_Normal(Policy):
    def __init__(self, basis: Basis, n_actions: int, sigma:Union[float, np.ndarray]=1.0, train_sigma:bool=True):
        self.basis = basis
        self.n_actions = n_actions
        self.n_inputs = basis.getNumFeatures()
        self.train_sigma = train_sigma
        self.basis = basis
        self.theta = np.zeros((self.n_inputs, n_actions), dtype=np.float64)
        self.std = np.ones((n_actions), np.float64) * sigma

        self.train_sigma = train_sigma
        self.num_params = self.theta.size + train_sigma * self.std.size

    def constrain_std(self):
        self.std = np.clip(self.std, 0.001, None)

    def get_action(self, obs:np.ndarray, stochastic:bool=True) -> Tuple[np.ndarray, float]:
        x = self.basis.encode(obs)

        mu, std = self.get_p(x)

        if stochastic:
            a = mu + np.random.normal(size=mu.shape) * std
            logp = self.logp(a, mu, std)
        else:
            a = mu
            logp = 0.

        return a, logp

    def get_num_params(self):
        return self.num_params

    def get_params(self):
        params = self.theta.flatten()
        if self.train_sigma:
            params = np.concatenate([params, self.std.flatten()])
        return params

    def add_to_params(self, grad):
        num_theta = self.theta.size
        self.theta += grad[:num_theta].reshape(self.theta.shape)
        if self.train_sigma:
            self.std += grad[num_theta:]
            self.constrain_std()

    def set_params(self, params):
        num_theta = self.theta.size
        self.theta = params[:num_theta].reshape(self.theta.shape)
        if self.train_sigma:
            self.std = params[num_theta:]
            self.constrain_std()

    def grad_logp(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        x = self.basis.encode(obs)
        theta = self.theta
        a = action.astype(np.float64)


        mu, std = self.get_p(x, theta)

        var = np.square(std)
        amu = (a - mu)

        gtheta = self.grad_logp_mu(x, (amu / var), theta).flatten()

        if self.train_sigma:
            gstd = self.grad_logp_std(amu, std)
            grad = np.concatenate([gtheta, gstd])
        else:
            grad = gtheta
        logp = self.logp(a, mu, std)

        return grad.flatten(), logp

    def logp(self, a, mu, std):
        var = np.square(std)
        log_std = np.log(std)
        a = a.astype(np.float32)
        logp = (-np.square(a - mu) / (2 * var)) - log_std - np.log(np.sqrt(2 * np.pi))
        return logp.sum()

    def grad_logp_std(self, amu, std):
        var = np.square(std)
        gstd = (1 / std) *(-1 + (np.square(amu) / var))
        grad = gstd

        return grad.flatten()

    def grad_logp_mu(self, x, gmu=1, theta=None):
        if theta is None:
            theta = self.theta
        if not isinstance(gmu, np.ndarray):
            gmu = np.array(gmu)

        gtheta = x.reshape(theta.shape[0], 1) * (gmu).reshape(1, self.n_actions)
        return gtheta.flatten()

    def get_p(self, x:np.ndarray, theta:Union[np.ndarray, None]=None)-> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(theta, np.ndarray):
            theta = self.theta
        mu = np.dot(x, theta)
        std = self.std

        return mu, std


class Linear_Dict(Policy):
    def __init__(self, policies:Dict[str, Policy]):
        self.policies = policies
        self.nparams = [pi.get_num_params() for k, pi in policies.items()]
        self.num_params = np.sum(self.nparams)

    def get_action(self, obs:np.ndarray, stochastic:bool=True) -> Tuple[Dict[str, Union[np.ndarray, int]], float]:
        acts = OrderedDict()
        logps = []
        for k, pi in self.policies.items():
            a, logp = pi.get_action(obs, stochastic)
            acts[k] = a
            logps.append(a)
        tlogp = float(np.sum(logps))

        return acts, tlogp

    def get_num_params(self):
        return self.num_params

    def get_params(self):
        params = np.concatenate([pi.get_params() for k, pi in self.policies.items()])

        return params

    def add_to_params(self, grad):
        nums = 0
        for k, pi in self.policies.items():
            num = pi.get_num_params()
            pi.add_to_params(grad[nums:nums+num])
            nums += num

    def set_params(self, params):
        nums = 0
        for k, pi in self.policies.items():
            num = pi.get_num_params()
            pi.set_params(params[nums:nums + num])
            nums += num

    def grad_logp(self, obs: np.ndarray, action: Dict[str, Dict[str, Union[np.ndarray, int]]]) -> Tuple[np.ndarray, float]:
        grads = []
        logps = []
        for k, act in action.items():
            grad, logp = self.policies[k].grad_logp(obs, act)
            grads.append(grad.flatten())
            logps.append(logp)

        grad = np.concatenate(grads)
        logp = float(np.sum(logps))

        return grad, logp

