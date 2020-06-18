# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:49:04 2019

@author: Luan
"""

# line search
import bracket_search as bs

from scipy import optimize

def linesearch(f,x,d):
    obj = lambda a: f(x+a*d)
    a,b = bs.bracket_minimum(obj)
    alpha = optimize.minimize_scalar(obj,[a,b],method = 'brent')
    return x+alpha.x*d

# teste

#print(linesearch(lambda x: x**3+4*x**2-20,1,3))

