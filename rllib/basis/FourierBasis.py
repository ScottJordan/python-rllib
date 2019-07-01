import numpy as np

from .basis import Basis

def increment_counter(counter, maxDigit):
    for i in list(range(len(counter)))[::-1]:
        counter[i] += 1
        if (counter[i] > maxDigit):
            counter[i] = 0
        else:
            break

class FourierBasis(Basis):
    def __init__(self, ranges, dorder, iorder, both=False):
        self.iorder = int(iorder)
        self.dorder = int(dorder)
        self.ranges = ranges.astype(np.float32)
        self.feat_range = (ranges[:, 1] - ranges[:, 0]).astype(np.float32)
        self.feat_range[self.feat_range == 0] = 1
        if both:
            self.feat_range = np.abs(ranges).max(axis=1)

        iTerms = iorder * ranges.shape[0]  # number independent terms
        dTerms = pow(dorder+1, ranges.shape[0])  # number of dependent
        oTerms = min(iorder, dorder) * ranges.shape[0]  # number of overlap terms
        self.num_features = int(iTerms + dTerms - oTerms)

        self.both = both
        self.num_input_features = int(ranges.shape[0])
        #print(self.num_input_features, iTerms, dTerms, self.num_features)
        #print("basis ", order, ranges.shape, self.num_features)
        self.C = np.zeros((self.num_features, ranges.shape[0]), dtype=np.float32)
        counter = np.zeros(ranges.shape[0])
        termCount = 0
        while termCount < dTerms:
            for i in range(ranges.shape[0]):
                self.C[termCount, i] = counter[i]
            increment_counter(counter, dorder)
            termCount += 1
        for i in range(ranges.shape[0]):
            for j in range(dorder+1, iorder+1):
                self.C[termCount, i] = j
                termCount += 1

        self.C = self.C.T * np.pi
        if both:
            self.num_features *= 2

    def encode(self, x):
        x = x.flatten()
        #scaled = (x.astype(np.float64) - self.ranges[:, 0]) / self.feat_range
        #if self.both:
        #    scaled = scaled * 2 - 1
        if self.both:
            scaled = x.astype(np.float64) / self.feat_range
        else:
            scaled = (x.astype(np.float64) - self.ranges[:, 0]) / self.feat_range
        dprod = np.dot(scaled, self.C)
        if self.both:
            basis = np.concatenate([np.cos(dprod), np.sin(dprod)])
        else:
            basis = np.cos(dprod)
        return basis

    def getNumFeatures(self):
        return self.num_features
