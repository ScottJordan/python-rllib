import numpy as np
import torch
import torch.nn as nn

from gym import spaces
from rllib.utils.netutils import to_tensor, to_numpy, get_flat_grad, get_flat_params, set_flat_grad, set_flat_params
from .critic import Critic

from typing import List, Tuple

class NetCritic(nn.Module, Critic):
    def predict(self, obs):
        raise NotImplementedError

    def grad(self, obs):
        v = self.predict(obs)
        grads = []

        if v.shape[0] == 1:
            self.zero_grad()
            v[0].backward()
            grads.append(get_flat_grad(self))
        else:
            for i in range(v.shape[0]):
                self.zero_grad()
                if i == v.shape[0] - 1:
                    v[i].backward(retain_graph=False)
                else:
                    v[i].backward(retain_graph=True)
                grads.append(get_flat_grad(self))

        grad = np.array(grads, dtype=np.float64)

        return grad

    def get_num_params(self):
        if self.num_params is not None:
            return self.num_params
        else:
            pp = 0
            for p in list(self.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            self.num_params = pp
            return self.num_params

    def get_params(self):
        return get_flat_params(self)

    def add_to_params(self, grad):
        params = get_flat_params(self)
        set_flat_params(self, params + grad)

    def set_params(self, params):
        set_flat_params(self, params)


    def pred_fn(self, theta, xt):
        old_params = get_flat_params(self)
        set_flat_params(self, theta)
        v = self.predict(xt)
        set_flat_params(old_params)
        return v

    def grad_fn(self, theta, xt):
        old_params = get_flat_params(self)
        set_flat_params(self, theta)
        grad = self.grad(xt)
        set_flat_params(old_params)
        return grad

class NetLinearVF(NetCritic):
    def __init__(self, in_dim):
        super(NetLinearVF, self).__init__()
        self.v = nn.Linear(in_dim, 1, bias=True)
        self.v.weight.data.copy_(torch.from_numpy(np.zeros(self.v.weight.shape)))


    def forward(self, x):
        return self.v(x)

    def predict(self, obs):
        if not torch.is_tensor(obs):
            obs = to_tensor(obs)
        v = self(obs)
        return v

class NetDeepVF(NetCritic):
    def __init__(self, base, out_layer):
        super(NetDeepVF, self).__init__()
        self.base = base
        self.out_layer = out_layer
        self.num_params = None

    def forward(self, x):
        if not isinstance(x,list) and x.dim() == 1:
            x = x.view(1, -1)
        if self.base is not None:
            out = self.base(x)
        else:
            out = x
        v = self.out_layer(out)
        return v

    def predict(self, obs):
        if not torch.is_tensor(obs):
            if isinstance(obs,list):
                obs = [to_tensor(i) for i in obs]
            else:
                obs = to_tensor(obs)
        v = self(obs)
        return v

class NetLinearQ(NetCritic):
    def __init__(self, in_dim, action_dim):
        super(NetLinearQ, self).__init__()
        self.action_dim = action_dim
        self.q = nn.Linear(in_dim, action_dim, bias=True)
        self.q.weight.data.copy_(torch.from_numpy(np.zeros(self.q.weight.shape)))


    def forward(self, x):
        return self.q(x)

    def predict(self, obs:Tuple):
        obs, acts = obs
        if not torch.is_tensor(obs):
            obs = to_tensor(obs)
        q = self(obs)
        #TODO I do not think this work for batch actions
        if acts is not None:
            return q[:, acts]
        else:
            return q

class NetDeepQ(NetCritic):
    def __init__(self, base, out_layer):
        super(NetDeepQ, self).__init__()
        self.base = base
        self.out_layer = out_layer
        self.num_params = None

    def forward(self, x):
        if not isinstance(x,list) and x.dim() == 1:
            x = x.view(1, -1)
        if self.base is not None:
            out = self.base(x)
        else:
            out = x
        q = self.out_layer(out)
        return q

    def predict(self, obs:Tuple):
        obs, acts = obs
        if not torch.is_tensor(obs):
            if isinstance(obs,list):
                obs = [to_tensor(i) for i in obs]
            else:
                obs = to_tensor(obs)
        q = self(obs)
        # TODO I do not think this work for batch actions
        if acts is not None:
            return q[:, acts]
        else:
            return q
