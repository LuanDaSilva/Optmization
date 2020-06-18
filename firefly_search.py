# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:48:42 2019

@author: Luan
"""

# atração dos vagalumes
import numpy as np

from rand_population_normal import rand_population_normal as rpn
from profiler import profile
#from scipy.stats import multivariate_normal

@profile
def firefly(f,population,k_max,beta=0.5,alpha=1e-4,brig = lambda r: np.exp(-1*r**2)):
    for k in range(k_max):
        #aux = []
        #a_new = 0
        for a in population:
            for b in population:
                if f(b) < f(a):
                    #print('a antigo: ',a)
                    r = np.linalg.norm(b-a)
                    a+= beta*brig(r)*(b-a)+alpha*np.random.randn(2)       
                    #print('a novo: ',a)
                    #aux.append(a)
        #population = np.array(aux)
        print(population[np.argmin([f(x) for x in population])])
    return population[np.argmin([f(x) for x in population])]

# teste

#f = lambda x: (x[0]-2)**2+x[1]**2
f = lambda x: (1-x[0])**2+(x[1]-x[0]**2)**2
m = 200
mu = [0,0]
S = np.eye(len(mu))
population = rpn(m,mu,S)
k_max = 150
print(f(firefly(f,population,k_max)))
                    
           