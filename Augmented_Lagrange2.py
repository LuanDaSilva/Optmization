# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 13:48:33 2020

@author: Luan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:53:54 2019

@author: Luan
"""
#import matplotlib.pyplot as plt
#from scipy import integrate
from scipy.optimize import minimize
#from powell import powell as minimize
import numpy as np


# Lagrange Aumentado - para restrições de igualdade

def augmented_lagrange_method(f,h,x,k_max,pho=1,gamma=2):
    l = np.zeros(len(h))
    for k in range(k_max):
        p = lambda x: f(x)+pho/2*sum([(hi(x)**2) for hi in h])-sum([l[i]*hi(x) for i,hi in enumerate(h)])
        #x = minimize(p,x,1e-8)
        x = minimize(p,x,method = 'TNC',tol = 1e-9).x  #minimize(p,x,10e-6)
        pho *=gamma
        l-=np.array([pho*hi(x) for hi in h])
        #print(x)
    return x

#f = lambda x: (1-x[0])**2+100*(x[1]-x[0]**2)**2
#f = lambda x: np.sin(x[1])*np.exp((1-np.cos(x[0]))**2)+np.cos(x[0])*np.exp((1-np.sin(x[1]))**2)+(x[0]-x[1])**2
#h1 = lambda x: (x[0]+5)**2-(x[1]+5)**2-25

#h = np.array([h1])

#k_max = 100

#x = np.array([1,1])

#print(augmented_lagrange_method(f,h,x,k_max))


# Remova o comentário daqui para baixo!
#f = lambda x,v: np.exp(-v[0]-v[1]*x-v[2]*x**2)
#f2= lambda x,v: f(x,v)*(x)
#f3 = lambda x,v: f(x,v)*(x)**2
#g = lambda x,v: f(x,v)*np.log(f(x,v))
#S = lambda v: integrate.fixed_quad(g,-3,3,args = (v,))[0]
#h1 = lambda v: integrate.fixed_quad(f,-3,3,args = (v,))[0]-1
#h2 = lambda v: integrate.fixed_quad(f2,-3,3,args = (v,))[0]-0
#h3 = lambda v: integrate.fixed_quad(f3,-3,3,args = (v,))[0]-1
#h = [h1,h2,h3]
#v1 = augmented_lagrange_method(S,h,[0,0,0],100)
#
#x = np.linspace(-10,10,1000)
#H = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-.5*x**2)
#plt.plot(x,f(x,v),'*g', label = 'MaxEnt')
#plt.plot(x,H(x),'r',label = 'Normal')
#plt.xlabel('Support')
#plt.ylabel('$f_Y$')
#plt.legend()