# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:26:57 2019

@author: Luan
"""
import numpy as np
from gradient_descent1 import Steepest_Descent as minimize

def interior_point_method(f,p,x,pho=1,y=2,tol=1e-5):
    delta = 10**10
    #print(delta)
    while delta>tol:
        #print(p(x))
        #po = p(x)/pho
        #print(po)
        x_new = minimize(lambda x:f(x)+p(x)/pho,x,tol=1e-8)
        print(x_new)
        delta = np.linalg.norm(x_new-x)
        print(delta)
        x=x_new
        pho*=2
    return x

#g1 = lambda x: (x[0]-5)**2+(x[1]-5)**2

g1 = lambda x: x[0]**2+x[1]**2-(1+ 0.2*np.cos(8*np.arctan(x[0]/x[1])))**2

g = [g1]
p = lambda x: -sum([1/gi(x) for gi in g])

f = lambda x: np.sin(x[1])*np.exp((1-np.cos(x[0]))**2)+np.cos(x[0])*np.exp((1-np.sin(x[1]))**2)+(x[0]-x[1])**2

#f = lambda x: 0.1*x[0]*x[1]

x0 = np.array([-10,-10])

print(interior_point_method(f,p,x0))