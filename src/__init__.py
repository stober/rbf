#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: __INIT__.PY
Date: Friday, June 15 2012
Description: Radial Basis Functions
"""

import numpy as np
import numpy.random as npr
import numpy.linalg as la

class RBF(object):

    def __init__(self, center, variance):
        self.center = center
        self.variance = variance

    def response(self, x):
        r = la.norm(self.center - x) # try inf norm?
        return np.exp( - (self.variance * r) ** 2 )

class RBFNetwork(object):

    def __init__(self, centers, variances):
        self.rbf = []
        self.size = len(centers)
        self.weights = np.ones(self.size + 1)
        for c,v in zip(centers,variances):
            self.rbf.append(RBF(c,v))

    def responses(self, x):
        return np.array([phi.response(x) for phi in self.rbf])

    def eval(self, x):
        r = np.ones(self.size + 1)
        r[:self.size] = self.responses(x)
        return np.sum(self.weights * r)

    def __repr__(self):
        return "RBF Network (%d functions)" % self.size

    def train(self, samples, targets):
        nsamples = len(samples)
        G = np.zeros((nsamples, self.size + 1))
        G[:,-1] = 1.0
        for (i,s) in enumerate(samples):
            G[i,:self.size] = self.responses(s)

        self.weights = np.dot(la.pinv(G), targets)



if __name__ == '__main__':

    pass
