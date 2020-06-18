# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:43:57 2019

@author: Luan
"""
import numpy as np
from scipy.stats import multivariate_normal
from profiler import profile
#a = rv.rvs(10)
#rv = multivariate_normal(np.mean(a,axis = 0),np.cov(a.T))
@profile
def cross_entropy_method(f,P,k_max,m = 300,m_elite=10):
    for k in range(1,k_max):
        samples = P.rvs(m)
        order = np.argsort([f(samples[i]) for i in range(m)])
        best_samples = np.array([samples[order[i]] for i in range(m_elite)])
        P = multivariate_normal(np.mean(best_samples,axis = 0),np.cov(best_samples.T))
    return P
#f = lambda x: (1-x[0])**2+(x[1]-x[0]**2)**2
f = lambda x:np.sin(x[1])*np.exp((1-np.cos(x[0]))**2)+np.cos(x[0])*np.exp((1-np.sin(x[1]))**2)+(x[0]-x[1])**2 #(x[1]-5.1/(4*np.pi**2)*x[0]**2+5/np.pi*x[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[0])+10

P = multivariate_normal([0.5, 0.5], [[2.0, 1.0], [1.0, 2.0]])
k_max = 100
print(cross_entropy_method(f,P,k_max).mean)
#v0 = [0.5,0.5]



        
        


