# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:42:18 2019

@author: Luan
"""
import numpy as np
from ghpkg import Jacobian as Jacobian
from Matthew_Davies_Alg import Matthew_Davies_Algorithm as MDA
from profiler import profile
from linesearch import linesearch as ls

@profile
def Gauss_Newton_algorithm(f,x0,tol = 1e-10):
    xk = x0
    n = (len(f))
    Fk =lambda x:sum([(f[i](x))**2 for i in range(n)])
    F_xant = Fk(xk)
    err = 10**10
    while err>tol:
        f_k = np.array([f[i](xk) for i in range(n)])
        J = Jacobian(f,xk,h = 1e-6)
        gk = 2*J.T.dot(f_k)
        Hk = np.array(2*J.T.dot(J))
        Lk,Dk = MDA(Hk)
        #print(Lk,Dk)
        yk = -Lk.dot(gk.T)
        #print(yk)
        dk = np.array((Lk.T.dot(np.linalg.inv(Dk)).dot(yk)))
        xk = ls(Fk,xk,dk.T[0])
        F_x = Fk(xk)
        err = abs(F_x-F_xant)
        F_xant = F_x
    return xk

# teste

f1 = lambda x: 3*x[0]-np.cos(x[1]*x[2]) #(x[0]-2)*(x[1]-1)
f2 = lambda x: x[0]**2-81*(x[1]+0.1)**2+np.sin(x[2])+1.06#(x[0]-10)**2
f3 = lambda x: np.exp(-x[0]*x[1])+20*x[2]+(10*np.pi-3)/3#x[1]+(x[0]-5)**2-3
f4 = lambda x: 100*(1-x[0])**2+(x[1]-x[0]**2)**2
F = np.array([f1,f2,f3,f4])
x0 = np.array([0.1,0.1,-0.1,1])

xkg = Gauss_Newton_algorithm(F,x0)
n = len(F)
f_kg = np.array([F[i](xkg) for i in range(n)])
print(xkg)
print(f_kg)

        
        
    
    
