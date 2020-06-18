# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:53:54 2019

@author: Luan
"""
from scipy import integrate
#from gradient_descent1 import Steepest_Descent as minimize
#from cross_entropy_method import cross_entropy_method as minimize
#from BFGS import BFGS as minimize
#from NelderMead import nelder_mead as minimize
#from Cyclic_coord_descent import cyclic_coordinate_descent as minimize
from powell import powell as minimize
import numpy as np


# Lagrange Aumentado - para restrições de igualdade

def augmented_lagrange_method(f,h,x,k_max,pho=1,gamma=2):
    l = np.zeros(len(h))
    for k in range(k_max):
        p = lambda x: f(x)+pho/2*sum([(hi(x)**2) for hi in h])-sum([l[i]*hi(x) for i,hi in enumerate(h)])
        #x = minimize(p,x,1e-8)
        x = minimize(p,x,10e-6)
        pho *=gamma
        l-=np.array([pho*hi(x) for hi in h])
        print(x)
    return x

#f = lambda x: (1-x[0])**2+100*(x[1]-x[0]**2)**2
#f = lambda x: np.sin(x[1])*np.exp((1-np.cos(x[0]))**2)+np.cos(x[0])*np.exp((1-np.sin(x[1]))**2)+(x[0]-x[1])**2
#h1 = lambda x: (x[0]+5)**2-(x[1]+5)**2-25

#h = np.array([h1])

#k_max = 100

#x = np.array([1,1])

#print(augmented_lagrange_method(f,h,x,k_max))
f = lambda x,v: np.exp(-v[0]-v[1]*x-v[2]*(x**2))
f2 = lambda x,v: x*f(x,v)
f3 = lambda x,v: f(x,v)*x**2
g = lambda x,v: f(x,v)*np.log(f(x,v))
S = lambda v: integrate.fixed_quad(g,-400,400,args = (v,))[0]
h1 = lambda v: integrate.fixed_quad(f,-400,400,args = (v,))[0]-1
h2 = lambda v: integrate.fixed_quad(f2,-400,400,args = (v,))[0]-50
h3 = lambda v: integrate.fixed_quad(f3,-400,400,args = (v,))[0]-100
h = [h1,h2,h3]
v = augmented_lagrange_method(S,h,[0,0,0],500)

x = np.linspace(-400,400,100)
#H = lambda x: 0.02*np.exp(-0.02*x)
plt.plot(x,f(x,v),'*g', label = 'MaxEnt')
#plt.plot(x,H(x),'r',label = 'Exponential')
plt.xlabel('Support')
plt.ylabel('$f_Y$')
plt.legend()

