# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:43:43 2019

@author: Luan
"""

import numpy as np

from rand_population_normal import rand_population_normal 
from rand_population_cauchy import rand_population_cauchy
#from rand_population_U import rand_population_U

#Selection Methods

def Truncation_Selection(k,y):
    p = np.argsort(y)
    return np.array([p[np.random.randint(low=0,high=k,size=2)] for i in y])
def Tournament_Selection(k,y):
    def getparent():
        p = np.random.permutation(len(y))
        val = p[np.argmin([y[i] for i in p[0:k]])]#p[np.argmin([y[i] for i in range(k)])]
        return val
    return np.array([[getparent(),getparent()] for i in y])
def RouletteWheelSelection(y):
    y = max(y)-y
    y = y/sum(y)
    #print(y)
    vec = np.random.choice(np.arange(0,len(y)),size = [len(y),2],p = y)
    return vec


# Crossover Methods

def SinglePoint_Crossover(a,b):
    #print(len(a))
    #np.random.seed(0)
    i = np.random.randint(0,len(a))
    return np.concatenate((a[0:i],b[i:]),axis=0)

def TwoPoint_Crossover(a,b):
    n = len(a)
    i,j = np.random.randint(low=0,high=n,size = 2)
    if i>j:
        (i,j) = (j,i)
    return np.concatenate((a[0:i],b[i:],a[j:n]))

def UniformCrossver(a,b):
    child = a
    for i in range(len(a)):
        if np.random.rand()<0.5:
            child[i] = b[i]
    return child
            
def InterpolationCrossover(w,a,b):
    return (1-w)*a+w*b


def GaussianMutation(sigma,child):
    return child+sigma*np.random.randn(len(child))


#child = np.array([SinglePoint_Crossover(population[p[0]],population[p[1]]) for p in pos])
def Genetic_Algorithm(f,k_max,population):
    y = [f(x) for x in population]
    for k in range(k_max):
        #pos = Truncation_Selection(10,y)
        pos = Tournament_Selection(5,y)
        #pos = RouletteWheelSelection(y)
        mom = np.array([population[p] for p in pos[:,0]])
        
        dad =  np.array([population[p] for p in pos[:,1]])
        
        #child = SinglePoint_Crossover(mom,dad)
        
        #child = TwoPoint_Crossover(mom,dad)
        
        child = UniformCrossver(mom,dad)
        
        #child = InterpolationCrossover(0.5,mom,dad)
        
        mutation = np.array([GaussianMutation(0.05,child) for child in child])
        
        population = mutation
        
        y = [f(x) for x in population]
        print('generation:{} - x: {}'.format(k,population[np.argmin(y)]))
        
    return population[np.argmin(y)]
    
    #print(x) 
    
# Set up

k_max = 100    
m = 10000

 #Normal population 
#

#Cauchy

mu = [0,0]

sigma = [1,1]

#population = rand_population_normal(m,mu,sigma)
population = rand_population_cauchy(m,mu,sigma)
#f = lambda x: np.linalg.norm(x)
#f = lambda x:(1-x[0])**2+(x[1]-x[0]**2)**2
f = lambda x: (x[0]-4)**2+(x[1]-2)**2
print(Genetic_Algorithm(f,k_max,population))


