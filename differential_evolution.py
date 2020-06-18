# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:36:35 2019

@author: Luan
"""
import numpy as np

from rand_population_normal import rand_population_normal as rpn

def differential_evolution(f,population,k_max,p=0.5,w=1):
    n,m = len(population[0]),len(population)
    for k in range(k_max):
        aux = []    
        for (k,x) in enumerate(population):
            r,s,t = np.random.choice(np.arange(m),3)
            a,b,c = population[r],population[s],population[t]
            z = a+w*(b-c)
            j = np.random.randint(0,n)
            x_new = [z[i] if (i==j or np.random.rand()<p) else x[i] for i in range(n)]
            if f(x_new)<f(x):
                x = x_new
            aux.append(x)
        population = np.array(aux)
    return population[np.argmin([f(x) for x in population])]



def Ackeley (x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    f = -a*np.exp(-b*np.sqrt(sum([val**2 for val in x])/d)) - np.exp(sum([np.cos(c*xi) for xi in x])/d) + a + np.exp(1)
    return f





# teste
m = 100
mu = [0,0]
S = np.eye(len(mu))
population = rpn(m,mu,S)
f = lambda x: (1-x[0])**2+(x[1]-x[0]**2)**2 #(x[1]-5.1/(4*np.pi**2)*x[0]**2+5/np.pi*x[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[0])+10
#f = lambda x: Ackeley(x)
k_max = 100
print(differential_evolution(f,population,k_max))
