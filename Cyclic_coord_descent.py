# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:38:54 2019

@author: Luan
"""

import numpy as np
from linesearch import linesearch as ls
# função já veio escalada!:)


def basis(i,n):
    d = np.zeros(n)
    d[i-1] = 1
    return d
#n = 2
#for i in range(1,n+1):
#    print(basis(i,n))

def cyclic_coordinate_descent(func,x0,tol):
    err,n = 10**100,len(x0)
    while (err)>tol:
        x = x0
        for i in range(1,n+1):
            d = basis(i,n)
            x0 = ls(func,x0,d)
        err = np.linalg.norm(x-x0,2)
        print(err)
        #x0 = x
    return x

#print(cyclic_coordinate_descent(func = lambda x,y:(1-x)**2+(y-x**2)**2,x0 = np.array([1,0]),tol = 1e-8))
#print(cyclic_coordinate_descent(func = lambda x:np.sin(x[0]+x[1])+(x[0]-x[1])**2-1.5*x[0]+2.5*x[1]+1,x0 = np.array([0,0]),tol = 1e-8))
    