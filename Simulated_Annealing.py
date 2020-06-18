# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:08:37 2019

@author: Luan
"""
import numpy as np
def simulated_annealing(f,x,t,T,k_max):
    print(x,t,T)
    c = 0.5
    t_k = lambda k: t/k
    y = f(x)
    x_best,y_best = x,y
    for k in range(0,k_max):
        x_new = x + (np.random.rand()-0.5)
        y_new = f(x_new)
        dy = y_new-y
        if dy<=0 or np.random.rand()< np.exp(-dy/t_k(k+1)):
            x,y = x_new,y_new
        if y_new < y_best:
            x_best,y_best = x_new,y_new
    return x_best


print(simulated_annealing(lambda x:x**2-np.cos(x),100,40,0.01,5000))