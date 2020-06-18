# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:05:33 2019

@author: Luan
"""
#import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import multivariate_normal

def rand_population_normal(m, mu, S):
    P = multivariate_normal(mu,S)
    return np.array(P.rvs(m))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
m = 10000
mu = [0,0]
S = np.eye(len(mu))

#R = rand_population_normal(m,mu,S)
#xs,ys,zs = R[:,0],R[:,1],R[:,2]
#
#ax.scatter(xs, ys, zs)
#ax.set_xlabel('X1')
#ax.set_ylabel('X2')
#ax.set_zlabel('X3')
