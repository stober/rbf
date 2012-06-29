#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_RBF.PY
Date: Friday, June 29 2012
Description: Test code for RBF networks.
"""

import numpy as np
import numpy.random as npr
import numpy.linalg as la
import pylab
from mpl_toolkits.mplot3d import Axes3D
from utils import rsme
from rbf import RBFNetwork

centers = npr.uniform(low=0,high=2 * np.pi, size=(20,2))
variances = np.ones(20) * 0.5
n = RBFNetwork(centers, variances)

# Training
points = npr.uniform(low=0,high=2*np.pi,size=(10000,2))
responses = np.sin(points[:,0]) + np.cos(points[:,1])
predictions = [n.eval(point) for point in points]
error = rsme(predictions,responses)
print "RSME prior to training: ", error

n.train(points,responses)

# Testing
points = npr.uniform(low=0,high=2*np.pi,size=(1000,2))
responses = np.sin(points[:,0]) + np.cos(points[:,1])
predictions = [n.eval(point) for point in points]
error = rsme(predictions, responses)
print "RSME after training: ", error

fig1 = pylab.figure(1)
ax1 = Axes3D(fig1)
ax1.scatter3D(points[:,0], points[:,1], responses)

fig2 = pylab.figure(2)
ax2 = Axes3D(fig2)
ax2.scatter3D(points[:,0], points[:,1], predictions)

pylab.show()
