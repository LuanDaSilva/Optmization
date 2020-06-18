# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:42:04 2019

@author: Luan
"""
from scipy import integrate
import numpy as np
#from Conjugate_Gradient_1 import Conjugate_Gradient as minimize
from gradient_descent1 import Steepest_Descent as minimize
#from gradient_descent_momentum import Steepest_Descent as minimize
#from NelderMead import nelder_mead as minimize
def penalty_function(g,h,x):
    p_count_g,p_count_h=0,0
    if g!=None:
        for gi in g:
            if gi(x)>0:
                p_count_g+=1
    if h!=None:
        for hi in h:
            if hi(x)!=0:
                p_count_h+=1
    return p_count_g+p_count_h
            
    
    
def penalty_method(f,g,h,x,k_max,pho=1,gamma = 2):
    for k in range(k_max):
        #p = (penalty_function(g,h,x))
        f_x = lambda x:f(x)+pho*penalty_function(g,h,x)
        x = minimize(f_x,x,tol=1e-9)
        pho*=gamma
        print(x)
        if pho*penalty_function(g,h,x) ==0:
            return x
        #print(x)#,pho*p)
    return x
        
# METHOD SETUP

# function

#f = lambda x: (1-x[0])**2+100*(x[1]-x[0]**2)**2
#f = lambda x: 0.1*x[0]*x[1]
#f = lambda x: np.sin(x[1])*np.exp((1-np.cos(x[0]))**2)+np.cos(x[0])*np.exp((1-np.sin(x[1]))**2)+(x[0]-x[1])**2
# restrições

#g1 = lambda x: x[0]**2+x[1]**2-(1+ 0.2*np.cos(8*np.arctan(x[0]/x[1])))**2 #
#g2 = lambda x: (x[0]-1)**3-x[1]+1
#g = [g1]
#h1 = lambda x: (x[0]-2)**2+(x[1]-2)**2
#h = None#[h1]

#x0 = np.array([-0.5,0.5])
#k_max = 150
#x = penalty_method(f,g,h,x0,k_max)
x0 = [1]
f = lambda x,v: np.exp(-v[0]*x)
g = lambda x,v: f(x,v)*np.log(f(x,v))
S = lambda v: integrate.quad(g,0,1,args = (v,))[0]
h1 = lambda v: integrate.quad(f,0,1,args = (v,))[0]-1
h = [h1]
x = penalty_method(S,None,h,x0,100)