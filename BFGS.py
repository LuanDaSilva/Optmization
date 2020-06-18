# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:28:05 2019

@author: Luan
"""
from ghpkg import gradient as grad
#from gradient import gradient as grad
import numpy as np
import linesearch as ls
#from profiler import profile

#@profile
def BFGS(func,x0,tol):
    i = 0
    Q = np.eye(len(x0))
    g_ant= grad(func,x0,1e-5)
    x_ant = x0
    x = ls.linesearch(func,x0,-Q.dot(g_ant))
    err = np.linalg.norm(x-x_ant,2)
    while err>tol:
        i+=1
        g =  grad(func,x,1e-5)
        if np.linalg.norm(g) == 0:
            return x
            break
        d = np.matrix(x-x_ant)
        x_ant = x
        y = np.matrix(g-g_ant)
        g_ant = g
        #print(g_ant)
        Q = np.matrix(Q)
        #Q = np.array(Q - (Q.dot(y.T.dot(y)).dot(Q))/y.dot(Q.dot(y.T))+d.dot(d.T)/d.dot(y.T))
        Q = np.array(Q+(y.dot(Q.dot(y.T))/d.dot(y.T))*(d.dot(d.T)/d.dot(y.T))-((d.T.dot(y)).dot(Q)+Q.dot(y.T.dot(d)))/d.dot(y.T)+(d.dot(d.T)/d.dot(y.T)))
        x = ls.linesearch(func,x,-Q.dot(g_ant))
        err = np.linalg.norm(x-x_ant,2)
        #print(x,i)
    return x

print(BFGS(func = lambda x:100*(1-x[0])**2+(x[1]-x[0]**2)**2,x0=np.array([0,0]),tol = 1e-8))
#print(BFGS(func = lambda x:(1-x[0])**2+(x[1]-x[0]**2)**2,x0=np.array([10,10]),tol = 1e-8))    
#print(BFGS(func = lambda x,y:(x**2-2)**2+(y-4)**2,x0=np.array([0,0]),tol = 1e-8))
