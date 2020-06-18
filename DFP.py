# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:43:00 2019

@author: Luan
"""
from ghpkg import gradient as grad
#from gradient import gradient as grad
import numpy as np
import linesearch as ls

def DFP(func,x0,tol):
    i = 0
    Q = np.eye(len(x0))
    g_ant= grad(func,x0,1e-4)
    if np.linalg.norm(g_ant) == 0:
        return x0
    x_ant = x0
    x = ls.linesearch(func,x0,-Q.dot(g_ant))
    #print(x)
    err = np.linalg.norm(x-x_ant,2)
    while err>tol:
        i+=1
        g =  grad(func,x,1e-4)
        #print(g)
        if np.linalg.norm(g) == 0:
            return x
            break
        d = np.matrix(x-x_ant)
        x_ant = x
        #print(x_ant)
        y = np.matrix(g-g_ant)
        g_ant = g
        Q = np.matrix(Q)
        Q = np.array(Q - (Q.dot(y.T.dot(y)).dot(Q))/y.dot(Q.dot(y.T))+d.dot(d.T)/d.dot(y.T))
        #print(Q)
        #print(-Q.dot(g_ant))
        x = ls.linesearch(func,x,-Q.dot(g_ant))
        err = np.linalg.norm(x-x_ant,2)
        #print(err)
    return x

#print(DFP(func = lambda x:x[0]**2+x[1]**2+x[2]**2,x0=np.array([1,1,1]),tol = 1e-8))
#print(DFP(func = lambda x: np.sin(x[1])*np.exp((1-np.cos(x[0]))**2)+np.cos(x[0])*np.exp((1-np.sin(x[1]))**2)+(x[0]-x[1])**2,x0=np.array([-5,-5]),tol = 1e-8))
#print(DFP(func = lambda x:100*(1-x[0])**2+(x[1]-x[0]**2)**2,x0=np.array([0,0]),tol = 1e-8))
#print(DFP(func = lambda x:(x[0])**2+(x[1])**2+(x[2])**2,x0=np.array([1,1,0]),tol = 1e-8))
