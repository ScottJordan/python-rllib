
import numpy as np
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
import torch

from .policies import Policy

from rllib.utils.netutils import to_tensor, to_numpy, get_flat_params, get_flat_grad, set_flat_params, set_flat_grad

class MLPBase(nn.Module):
    def __init__(self, obs_space, dims, act_fn=nn.LeakyReLU(), ranges=None):
        super(MLPBase, self).__init__()
        self.obs_space = obs_space
        self.dims = dims
        self.act_fn = act_fn


        in_dim = obs_space.low.shape[0]
        in_dims = [in_dim] + dims[:-1]
        out_dims = dims
        self.ranges = ranges
        if ranges is not None: # scales input to be in 1/max(|x|) for each dim.
            # self.feat_range = (ranges[:, 1] - ranges[:, 0]).astype(np.float64) # scales to be [0,1]
            self.feat_range = np.abs(ranges).max(axis=1)
            self.feat_range[self.feat_range==0] = 1
            self.feat_range = to_tensor(self.feat_range, requires_grad=False)
            self.ranges = to_tensor(self.ranges, requires_grad=False)
        self.layers = nn.ModuleList(nn.Linear(idim, odim) for idim, odim in zip(in_dims, out_dims))

    def forward(self, x):
        if self.ranges is not None:
            # x = (x - self.ranges[:, 0]) / self.feat_range # for  [0, 1] scaling
            x = x / self.feat_range
        for layer in self.layers:
            x = self.act_fn(layer(x))

        return x

class NetSoftmax(nn.Module):
    def __init__(self, in_dim, act_dim):
        super(NetSoftmax, self).__init__()
        self.in_dim = in_dim
        self.act_dim = act_dim
        self.layer = nn.Linear(in_dim, act_dim, bias=True)

    def forward(self, x):
        p = F.softmax(self.layer(x), dim=1)
        # p = F.softmax(x,dim=1)
        return D.Categorical(p)


class NetNormal(nn.Module):
    def __init__(self, in_dim, act_dim, sigma, train_sigma=True):
        super(NetNormal, self).__init__()
        self.in_dim  = in_dim
        self.act_dim = act_dim
        self.mulayer = nn.Linear(in_dim, act_dim, bias=True)
        if train_sigma:
            self.std = nn.Parameter(torch.ones(act_dim) * sigma)
        else:
            self.std = torch.ones(act_dim) * sigma
        self.mulayer.weight.data.fill_(0.00)
        self.mulayer.bias.data.fill_(0.00)

    def constrain_std(self):
        small = self.std.data < 0.001
        if small.sum() > 0:
            self.std.data[small] = 0.001

    def forward(self, x):
        mu = self.mulayer(x)
        return D.Normal(mu, self.std.expand_as(mu))


class NetPolicy(nn.Module, Policy):
    def __init__(self, base, out_layer, numpy_action=False):
        super(NetPolicy, self).__init__()
        self.base = base
        self.out_layer = out_layer
        self.num_params = None
        self.numpy_action = numpy_action

    def forward(self, x):
        if not isinstance(x,list) and x.dim() == 1:
            x = x.view(1, -1)
        out = self.base(x)
        dist = self.out_layer(out)
        return dist

    def constrain_std(self):
        if isinstance(self.out_layer, NetNormal):
            self.out_layer.constrain_std()

    def get_action(self, obs, stochastic=True):
        if self.numpy_action:
            if not isinstance(obs, torch.Tensor):
                with torch.no_grad():
                    obs = to_tensor(obs)

            return self.get_action_numpy(obs, stochastic)
        else:
            if not isinstance(obs, torch.Tensor):
                if isinstance(obs,list):
                    obs = [to_tensor(ob) for ob in obs]
                else:
                    obs = to_tensor(obs)
            return self.get_action_torch(obs, stochastic)

    def get_action_numpy(self, obs, stochastic=True):
        x = obs
        with torch.no_grad():
            x = to_tensor(x).float()
            dist = self(x)

        if stochastic:
            #a = np.array([np.random.normal(loc=mu[i, :], scale=std[i, :]) for i in range(mu.shape[0])]).reshape(-1, mu.shape[1])
            action = dist.sample()
            logp = to_numpy(dist.log_prob(action).sum(dim=-1))
            a = to_numpy(action)
        else:
            if isinstance(dist, D.Categorical):
                a = np.argmax(to_numpy(dist.probs))
            elif isinstance(dist, D.Normal):
                a = to_numpy(dist.loc)
            else:
                raise Exception("dist type not recognized: "+str(type(dist)))
            logp = np.zeros_like(a).sum(axis=-1)

        if len(logp.shape) == 0:
            logp = np.array([logp])
        if len(a.shape) == 0:
            a = np.array([a])

        return a, logp

    def get_action_torch(self, obs, stochastic=True):
        x = obs

        # x = to_tensor(x)
        dist = self(x)

        if stochastic:
            #a = np.array([np.random.normal(loc=mu[i, :], scale=std[i, :]) for i in range(mu.shape[0])]).reshape(-1, mu.shape[1])
            action = dist.sample()
            logp = dist.log_prob(action).view(-1,1)
            a = action
        else:
            if isinstance(dist, D.Categorical):
                _, a = torch.max(dist.probs)#np.argmax(to_numpy(dist.probs))
            elif isinstance(dist, D.Normal):
                a = dist.loc #to_numpy(dist.loc)
            else:
                raise Exception("dist type not recognized: "+str(type(dist)))
            logp = torch.zeros_like(a)

        return a, logp

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

    def grad_logp(self, obs, action):
        x = to_tensor(obs)
        actions = to_tensor(action)

        dist = self(x)
        logp = dist.log_prob(actions)
        grads = []
        lp = logp.sum(dim=-1)
        if lp.shape[0] == 1:
            self.zero_grad()
            lp[0].backward()
            grads.append(get_flat_grad(self))
        else:
            for i in range(lp.shape[0]):
                self.zero_grad()
                if i == lp.shape[0] - 1:
                    lp[i].backward(retain_graph=False)
                else:
                    lp[i].backward(retain_graph=True)
                grads.append(get_flat_grad(self))

        grad = np.array(grads, dtype=np.float64)
        logp = to_numpy(lp).reshape(-1, 1)

        return logp, grad
