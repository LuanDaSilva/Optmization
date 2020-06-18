# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 22:06:26 2020

@author: Luan
"""
from scipy import integrate
import numpy as np
from Augmented_Lagrange2 import augmented_lagrange_method as ALM
def f_X(g):
   return lambda x,v: np.exp(sum([-v[i]*g[i](x) for i in range(len(g))]))

def moment(f,g,supp):
    mom = lambda v:integrate.fixed_quad(lambda x,v:f(x,v)*g(x),supp[0],supp[1],args = (v,))[0]
    return mom
    
def MaxEnt(supp,g_X,restrictions):
    f = f_X(g_X)
    S = lambda v: integrate.fixed_quad(lambda x,v: f(x,v)*np.log(f(x,v)),supp[0],supp[1],args = (v,))[0]
    moms = [moment(f,gi_X,supp) for gi_X in g_X]
    h =  [lambda v: moms[i](v)-restrictions[i] for i in range(len(restrictions))]
    v = ALM(S,h,np.zeros(len(restrictions)),100)
    return f,v

    

#import matplotlib.pyplot as plt


#Exemplo 1: Normal
#supp = (-3,3)   
#g_X = [lambda x:1,
#     lambda x: x,
#     lambda x: x**2]
#restrictions = [1, 0, 1]
#f,v = MaxEnt(supp,g_X,restrictions)
#x = np.linspace(supp[0],supp[1],100)
#H = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-.5*x**2)


#Exemplo 2: Uniforme 
#supp = (0,1)  
#restrictions = [1]
#g_X = [lambda x:1]
#x = np.linspace(supp[0],supp[1],100)
#H = lambda x: 1/(supp[1]-supp[0])
#f,v = MaxEnt(supp,g_X,restrictions)
#vec = np.array([f(xi,v) for xi in x])
#vec2 = np.array([H(xi) for xi in x])
#plt.plot(x,vec2-vec) # Erro
    
#Exemplo 3: Exponencial
    
supp = (-10,10)
g_X = [lambda x: 1, lambda x: x]
restrictions = [1,2]
f,v = MaxEnt(supp,g_X,restrictions)
x = np.linspace(supp[0],supp[1],1000)
plt.plot(x,f(x,v))




# Teste
#f_X = lambda x,v: 1/10 # Teste
#g = [lambda x:1,
#     lambda x: x,
#     lambda x: (x-moment(f_X,lambda x: x,supp)(0))**2,
#     lambda x: x**3,
#     lambda x: x**4]
#moms = lambda v:[moment(f_X,gi,supp)(v) for gi in g]
#func = lambda x,v: np.exp(-sum([v[i]*g[i](x) for i in range(len(g))]))
#restriction = [1.0, 5.0, 8.333333333333329, 249.9999999999999, 1999.9999999999989]

#f = f_X(g)