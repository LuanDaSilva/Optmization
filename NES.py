# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:20:25 2019

@author: Luan
"""
from scipy.stats import multivariate_normal
from scipy import optimize
import bracket_search as bs
import numpy as np
from Matthew_Davies_Alg import Matthew_Davies_Algorithm as MDA
from profiler import profile

def grads(x,mu,sigma):
    #aux1 = np.matrix(x-mu)
    d_mu = np.linalg.inv(sigma).dot(x-mu)
    aux1 = np.matrix(x-mu)
    aux2 = aux1.T.dot(aux1)
    d_sig = 0.5*np.linalg.inv(sigma).dot(aux2.dot(np.linalg.inv(sigma)))-0.5*np.linalg.inv(sigma)
    return(d_mu,d_sig)
    
#def Fisher(x,mu,sigma):
#    F_mu = grads(x,mu,sigma)[0].dot(grads(x,mu,sigma)[0].T)
#    F_sigma = grads(x,mu,sigma)[1].dot(grads(x,mu,sigma)[1].T)
#    return (F_mu,F_sigma)
def ls(f,x,d):
    obj = lambda a: f(x+a*d)
    a,b = bs.bracket_minimum(obj)
    alpha = optimize.minimize_scalar(obj,[a,b],method = 'brent')
    return alpha.x

@profile
def natural_evolution_strategies(f,P,k_max,m=1000):
    mu,sigma = P.mean,P.cov
    beta = 10**10
    bk = 1e-1
    vk = np.zeros(len(mu))
    for k in range(k_max):
        samples = P.rvs(m) 
        #grad = np.sum([grads(x+vk*bk,mu,sigma)[0]*f(x) for x in samples],axis=0)/m
        grad = np.sum([grads(x,mu,sigma)[0]*f(x) for x in samples],axis=0)/m
        a = ls(f,mu,-grad)
        vk = bk*vk-a*grad
        #mu+=vk
        mu-= a*grad
        #sigma-=a*np.sum([grads(x+vk*bk,mu,sigma)[1]*f(x) for x in samples],axis = 0)/m
        sigma-=a*np.sum([grads(x,mu,sigma)[1]*f(x) for x in samples],axis = 0)/m
#        if not np.all(np.linalg.eigvals(sigma) > 0):
#            sigma = (sigma+beta*np.eye(2,2))/(1+beta)
        L,D = MDA(sigma)
        sigma = L.T.dot(np.linalg.inv(D)).dot(L) # Método de Matthew-Davies!
        P = multivariate_normal(mu,sigma)
#        print('k = {}, x: {}'.format(k,P.mean))
    return P
        
#x = [1,1]
#G = grads(x,np.array([0.5, 0.5]), np.array([[2.0, 1.0], [1.0, 2.0]]))[1]

#mu,sigma = np.array([0.5, 0.5]), np.array([[2.0, 1.0], [1.0, 2.0]])
#x = np.array([1,1])
#grads(x,mu,sigma)[1]
P = multivariate_normal([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]])  
#f = lambda x: (x[1]-5.1/(4*np.pi**2)*x[0]**2+5/np.pi*x[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[0])+10
f = lambda x:-np.exp(-(x[0]*x[1] - 1.5)**2 -(x[1]-1.5)**2)
#f = lambda x:100*(1-x[0])**2+(x[1]-x[0]**2)**2
#f = lambda x:x[0]**2+x[1]**2
k_max=40
#var = P.rvs(100)
print(natural_evolution_strategies(f,P,k_max).mean)


  






        
        
        
        
        
        
        