import numpy as np
from gym import spaces

from typing import List, Union, Tuple, Dict

class Basis(object):
    def encode(self, x: np.ndarray)->np.ndarray:
        raise NotImplementedError

    def getNumFeatures(self) -> int:
        raise NotImplementedError

class PassThroughBasis(Basis):
    def __init__(self, num_features: int, ranges=None, max_norm:bool=False, add_const:bool=True):
        self.add_const = add_const
        self.num_features = int(num_features)
        if add_const:
            self.num_features +=1
        self.num_input_features = int(num_features)
        self.ranges = ranges
        self.max_norm = max_norm
        if ranges is not None:
            if max_norm:
                self.feat_range = np.abs(ranges).max(axis=1).astype(np.float64)
            else:
                self.feat_range = (ranges[:, 1] - ranges[:, 0]).astype(np.float64)

    def encode(self, x: np.ndarray)->np.ndarray:
        x = x.flatten()
        if self.ranges is not None:
            if self.max_norm:
                x = x.astype(np.float64) / self.feat_range
            else:
                x = (x.astype(np.float64) - self.ranges[:, 0]) / self.feat_range
        if self.add_const:
            const = np.ones(1, dtype=x.dtype)
            x = np.concatenate((x, const))

        basis = np.array(x)
        return basis

    def getNumFeatures(self):
        return self.num_features


class LinearDiscreteBasis(Basis):
    def __init__(self, obs_space:spaces.Discrete):
        self.num_features = obs_space.n

    def encode(self, x: int)-> np.ndarray:
        s = np.zeros((self.num_features,), dtype=np.float64)
        s[x] = 1
        return s

    def getNumFeatures(self) -> int:
        return self.num_features

class ConcatBasis(Basis):
    def __init__(self, basis:List[Basis], index_map):
        self.index_map = index_map
        self.basis = basis
        self.num_input_features = int(np.concatenate(index_map, axis=0).shape[0])
        self.num_features = int(np.sum([b.getNumBasisFunctions() for b in basis]))

    def encode(self, x):
        res = [self.basis[i].encode(x[:, self.index_map[i]]) for i in range(len(self.basis))]
        res = np.concatenate(res, axis=-1)

        return res

    def getNumBasisFeatures(self):
        return self.num_features
