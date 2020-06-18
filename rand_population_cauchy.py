# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:19:09 2019

@author: Luan
"""
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import cauchy

def rand_population_cauchy(m,mu,sigma):
    n = len(mu)
    Dist = [cauchy.rvs(loc = mu[i],scale = sigma[i],size = m) for i in range(n)]
    return np.array(Dist).T

# teste
    
#mu = np.array([0,0])
#sigma = np.array([1,1])
#m = 1000
#Q = rand_population_cauchy(m,mu,sigma)
#Q = Q.T
#plt.scatter(Q[:,0],Q[:,1])
##plt.xlabel('x1')
##plt.ylabel('x2')




