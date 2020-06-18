# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:28:47 2019

@author: Luan
"""
import numpy as np
from linesearch import linesearch as ls

def basis(i,n):
    d = np.zeros(n)
    d[i-1] = 1
    return d

def powell(f,x0,tol):
    err,n = 10**100,len(x0)
    U = [basis(i,n) for i in range(1,n+1)]
    #print(U)
    while err>tol:
        x = x0
        for i in range(0,n):
            d = U[i]
            x = ls(f,x,d)
        for i in range(0,n-1):
            U[i] = U[i+1]
        U[n-1] = d = x-x0
        x = ls(f,x,d)
        err = np.linalg.norm(x-x0,2)
        x0 = x
    return x0

#print(powell(lambda x:(1-x[0])**2+(x[1]-x[0]**2)**2,np.array([10,10]),1e-8))
            
    