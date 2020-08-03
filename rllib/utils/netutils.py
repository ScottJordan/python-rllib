import torch
import numpy as np
import os


USE_CUDA = os.environ.get("USE_GPU", str(False)).lower() == "true" and torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def logRand(min, max, size=1):
    return np.exp(np.random.uniform(np.log(min), np.log(max), size=size))

def to_numpy(var):
    if torch.is_tensor(var):
        if var.is_cuda:
            var = var.detach().cpu()
        return var.detach().numpy().astype(np.float32)
    elif isinstance(var, np.ndarray):
        return var
    else:
        raise TypeError("Unknown data type:", type(var))


def to_tensor(x, requires_grad=False, dtype=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if (torch.is_tensor(x)):
        x.requires_grad = requires_grad
        if dtype == torch.float64:
            x.double()
        elif dtype == torch.float32:
            x.float()
        elif dtype == torch.int32:
            x.int()
        elif dtype == torch.int64:
            x.long()
        if USE_CUDA:
            return x.cuda()
        else:
            return x

        # elif dtype == torch.cuda.float64:
        #     return x.double().cuda()
        # elif dtype == torch.cuda.float32:
        #     return x.float().cuda()
        # elif dtype == torch.cuda.int32:
        #     return x.int().cuda()
        # elif dtype == torch.cuda.int64:
        #     return x.long().cuda()
        # else:
        #     raise TypeError("Type {0} not understood by to_tensor".format(dtype))

    else:
        raise TypeError("variable with type '{0}' can not be made into a tensor. Requires numpy array, torch.Tensor".format(type(x)))

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def grad_update(model, grads, alpha):
    for param, grad in zip(model.parameters(), grads):
        param.data.copy_(param.data + alpha * grad.data)

def ordered_params(model):
    # for x in sorted(model.named_parameters(), key=lambda x: x[0]):
    #     print(x[0])
    namedparams = sorted(model.named_parameters(), key=lambda x: x[0])
    return [x[1] for x in namedparams]


def get_flat_params(model):
    params = ordered_params(model)
    if len(params) > 0:
        return to_numpy(torch.cat([param.data.view(-1) for param in params]))
    else:
        return to_numpy(torch.zeros(1))

def get_flat_torch_grad(model):
    params = ordered_params(model)
    if len(params) > 0:
        return torch.cat([param.grad.data.view(-1) for param in params])
    else:
        return torch.zeros(1)

def get_flat_grad(model):
    #xp = chain.xp
    params = ordered_params(model)
    if len(params) > 0:
        return to_numpy(torch.cat([param.grad.data.view(-1) for param in params]))
    else:
        return to_numpy(torch.zeros(1))


def set_flat_params(model, flat_params):
    offset = 0
    flat_params = torch.from_numpy(flat_params.flatten())
    for param in ordered_params(model):
        numel = param.numel()
        param.data.copy_(flat_params[offset:offset + numel].view(param.data.shape))
        offset += numel


def set_flat_grad(model, flat_grad):
    offset = 0
    flat_grad = torch.from_numpy(flat_grad)
    for param in ordered_params(model):
        numel = param.grad.numel()
        param.grad.data.copy_(flat_grad[offset:offset + numel].view(param.grad.shape))
        offset += numel
