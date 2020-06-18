# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:09:06 2019

@author: Luan
"""

# Categorical(p)
#A *Categorical distribution* is parameterized by a probability vector `p` (of length `K`).
#```math
#P(X = k) = p[k]  \\quad \\text{for } k = 1, 2, \\ldots, K.

import numpy as np

#def Categorical(y):
#    pi = y/np.linalg.norm(y)
#    Ui = np.random.rand(len(pi))
#    return max([np.log(pi[i]/(1-pi[i]))-np.log(np.log(1/Ui[i])) for i in range(len(pi))])
#    
#def RouletteWheelSelection(y):
#    return [[Categorical(y),Categorical(y)] for i in y]

def RouletteWheelSelection(y):
    y_hat = max(y)-y
    pi = y_hat/np.linalg.norm(y_hat)
    val = [i for i,p in enumerate(pi)]
    print(val)
    return pi
        
y = lambda x:x**2
x = np.array([1,0.5,2,-1.5])
f = [y(x_s) for x_s in x] 
#print(Categorical(y))
print(RouletteWheelSelection(f)) 
    
    
    
    