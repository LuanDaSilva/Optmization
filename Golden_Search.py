# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:57:40 2019

@author: Luan
"""
import numpy as np
# Golden Search

def Golden_Search(func,xl,xu,tol):
    K = (1+np.sqrt(5))/2
    I1 = xu-xl
    Ik = I1/K
    xa = xu-Ik
    xb = xl+Ik
    aux = Ik
    while Ik>tol:
        Ik = aux/K
        aux = Ik
        if func(xa)>=func(xb):
            xl = xa
            xu = xu
            xa = xb
            xb = xl+Ik
        else:
            xl = xl
            xu = xb
            xb = xa
            xa = xu-Ik
    if func(xa)>func(xb):
        xa = 0.5*(xb+xu)
    elif func(xa)<func(xb):
        xa = 0.5*(xa+xb)
    else:
        xa = 0.5*(xl+xa)
    return xa

#print(Golden_Search(lambda x: x**3-10*x**2-10,0,6,1e-5))
            