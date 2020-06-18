# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:09:54 2019

@author: Luan
"""
from ghpkg import gradient as grad
import linesearch as ls
import numpy as np
def Conjugate_Gradient(f,x0,tol):
    g = grad(f,x0,1e-5)
    d = -g
    #d = d/np.linalg.norm(d,2)
    xk = ls.linesearch(f,x0,d)
    g_ant = g
    xk_ant = xk
    err = abs(f(xk_ant))
    while err>tol:
        g = grad(f,xk,1e-5)
        b = max(0,g.dot(g-g_ant)/(g_ant.dot(g_ant)))
        d = -g+b*d
        #d = d/np.linalg.norm(d,2)
        xk = ls.linesearch(f,xk,d)
        print(xk)
        err = abs(f(xk)-f(xk_ant))
        xk_ant=xk
        g_ant = g
    return xk
    
    

#print(Conjugate_Gradient(f = lambda x: 0.1*x[0]*x[1],x0 = [0,0],tol = 1e-8)) # teste com Rosebrook
#print(Conjugate_Gradient(f = lambda x,y: (x-2*y**2)**2+(5*y-20)**2,x0 = [0.5,0],tol = 1e-8))
    
   
    
    
    
    
    
    