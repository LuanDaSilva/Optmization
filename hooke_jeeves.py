# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:48:52 2019

@author: Luan
"""
import numpy as np

def basis(i,n):
    d = np.zeros(n)
    d[i-1] = 1
    return d

def hooke_jeeves(func,x0,tol,a = 0.5,gamma = 0.5):
    y0,n = func(x0),len(x0)
    while a>tol:
        improved = False
        x_best,y_best = x0,y0
        for i in range(1,n+1):
            for sgn in [-1,1]:
                x = x0+sgn*a*basis(i,n)
                y = func(x)
                if y<y_best:
                    x_best,y_best,improved = x,y,True
        x0,y0 = x_best,y_best
        if not improved:
            a *= gamma
        #print(x)
    return x

print(hooke_jeeves(lambda x:(1-x[0])**2+(x[1]-x[0]**2)**2,np.array([10,10]),1e-8))
                