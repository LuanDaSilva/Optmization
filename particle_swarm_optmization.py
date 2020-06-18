# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:33:12 2019

@author: Luan
"""

# enxame de partículas
from profiler import profile
import numpy as np
from rand_population_normal import rand_population_normal as rpn


@profile
def particle_swarm_optimization(f,population,k_max,w=1,c1=1,c2=1,tol=1e-10):
    n = len(population[0])
    v = np.zeros(n)
    x_best,y_best = population[0],10**10
    aux = []
    for x in population:
        y = f(x)
        if y<y_best:x_best,y_best = x,y
       
    for k in range(k_max):
        for x in population:
            r1,r2 = np.random.rand(2)
            x+=v
            v = w*v+c1*r1*(x_best-x)+c2*r2*(x_best-x)
            y = f(x)
            if y<y_best: x_best,y_best = x,y
            if  y<f(x_best):x_best = x
            aux.append(x_best)
            #print(x_best)
        population = np.array(aux)
        if np.linalg.norm(population[np.argmin([f(x) for x in population])])<tol: return population[np.argmin([f(x) for x in population])]
            
    return population[np.argmin([f(x) for x in population])]

def Ackeley (x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    f = -a*np.exp(-b*np.sqrt(sum([val**2 for val in x])/d)) - np.exp(sum([np.cos(c*xi) for xi in x])/d) + a + np.exp(1)
    return f

# teste
    
f = lambda x: x[0]**2+x[1]**2
#f = lambda x: Ackeley(x)
m = 100
mu = [0,0]
S = np.eye(len(mu))
population = rpn(m,mu,S)
k_max = 10
#print(particle_swarm_optimization(f,population,k_max))
            