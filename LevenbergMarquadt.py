# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 10:20:34 2019

@author: Luan
"""

# Levenberg-Marquadt

import numpy as np
from ghpkg import Jacobian as Jacobian
from Matthew_Davies_Alg import Matthew_Davies_Algorithm as MDA

from linesearch import linesearch as ls


def Levenberg_Marquadt(f,x0,l,tol = 1e-12,l_upper=10**10,l_lower=1e-12):
    xk = x0
    n = (len(f))
    Fk =lambda x:sum([(f[i](x))**2 for i in range(n)])
    F_xant = Fk(xk)
    err,err_ant = 10**10,10**10
    while err>tol and l<=l_upper and l>=l_lower:
        f_k = np.array([f[i](xk) for i in range(n)])
        J = Jacobian(f,xk,h = 1e-6)
        gk = 2*J.T.dot(f_k)
        gk = gk/np.linalg.norm(gk)
        Hk = np.array(2*J.T.dot(J))
        Hk_hat = (Hk+l*np.diag(np.diag(Hk)))
        Lk,Dk = MDA(Hk_hat)
        yk = -Lk.dot(gk.T)
        dk = np.array((Lk.T.dot(np.linalg.inv(Dk)).dot(yk)))
        xk = ls(Fk,xk,dk.T[0])
        F_x = Fk(xk)
        err = abs(F_x-F_xant)
        if err_ant>=err:
            l*=0.1
        else:
            l*=10
        F_xant = F_x
        err_ant = err
        #print('xk = {}\n'.format(xk),'\nl = {}'.format(l))
    return xk

f1 = lambda x: 3*x[0]-np.cos(x[1]*x[2]) #(x[0]-2)*(x[1]-1)
f2 = lambda x: x[0]**2-81*(x[1]+0.1)**2+np.sin(x[2])+1.06 #(x[0]-10)**2
f3 = lambda x: np.exp(-x[0]*x[1])+20*x[2]+(10*np.pi-3)/3 #x[1]+(x[0]-5)**2-3
f4 = lambda x: 100*(1-x[0])**2+(x[1]-x[0]**2)**2
F = np.array([f1,f2,f3,f4])
x0 = np.array([0.1,0.1,-0.1,1])
l = 1e-1
xk = Levenberg_Marquadt(F,x0,l)
print(xk)
n = len(F)
f_k = np.array(([F[i](xk) for i in range(n)]))
print(f_k)

