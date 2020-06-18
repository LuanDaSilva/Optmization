# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:38:30 2019

@author: Luan
"""

# Cuckoo Search
import numpy as np
from rand_population_normal import rand_population_normal as  rpn
from rand_population_cauchy import rand_population_cauchy as  rpc

def cuckoo_search(f,population,k_max,p_a=0.1):
    m,n = len(population),len(population[0])
    a = round(m*p_a)
    for k in range(k_max):
        i,j = np.random.randint(m),np.random.randint(m)
        x = population[j]+np.random.standard_cauchy(n)
        y = f(x)
        if y<f(population[i]):
            population[i] = x
            y = f(population[i])
        #p = np.argsort(population)[::-1]
        p = np.argsort([f(x) for x in population])[::-1]
        for i in range(a):
            j = np.random.randint(low=0,high=m-a)+a
            population[p[i]] = population[p[j]]+np.random.standard_cauchy(n)
            #y = f(population[p[i]])
        print(population[np.argmin([f(x) for x in population])])
    return population[np.argmin([f(x) for x in population])]

# teste
    
f = lambda x: (x[0]-2)**2+(x[1]-2)**2#(1-x[0])**2+(x[1]-x[0]**2)**2#

mu = [0,0]
S = np.eye(len(mu))
m = 1000000
#population = rpn(m,mu,S)
#m = 10000
#mu = np.array([0,0])
sigma = np.array([1,1])
population = rpc(m,mu,sigma)

k_max = 50
print(f(cuckoo_search(f,population,k_max)))

