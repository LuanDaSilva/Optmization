# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:15:32 2019

@author: Luan
"""

import numpy as np
from scipy import optimize
import Golden_Search as gs
import bracket_search as bs
from strong_backtracking import strong_backtracking as sb
import BLS as bls
from ghpkg import gradient as gradient
from profiler import profile
#import Golden_Search as gs
#def gradient(func,x1,x2,h):
#    grad = []
#    grad.append((func(x1+h,x2)-func(x1-h,x2))/(2*h))
#    grad.append((func(x1,x2+h)-func(x1,x2-h))/(2*h))
#    #print(grad)
#    return np.array(grad)

@profile
def Steepest_Descent(func,x0,tol):
    i = 0
    bk = 1e-1
    vk = np.array([0,0])
    vk = np.zeros(len(x0))
    #dk = gradient(func,x0,1e-4)
    dk = gradient(func,x0+bk*vk,1e-4) # Momento de Nesterov
    dk = dk/np.linalg.norm(dk,2)
    alpha_k = sb(func,x0,d=-dk)
    vk = bk*vk-alpha_k*dk
    xk = x0+vk
    #xk = x0-alpha_k*dk
    while np.linalg.norm(dk*alpha_k,2)>tol:
        i+=1
        dk = gradient(func,xk,1e-4)
        dk = gradient(func,xk+bk*vk,1e-4) # Momento de Nesterov
        dk = dk/np.linalg.norm(dk,2)
        alpha_k = sb(func,xk,d = -dk)
        vk = bk*vk-alpha_k*dk
        xk = xk+vk
        #xk = xk-alpha_k*dk
        #print(xk,'número de iterações: {}'.format(i))
    return xk

print(Steepest_Descent(func = lambda x:100*(1-x[0])**2+(x[1]-x[0]**2)**2,x0=np.array([0.5,0]),tol = 1e-8))

#print(Steepest_Descent(func = lambda x:100*(1-x[0])**2+(x[1]-x[0]**2)**2,x0=np.array([0.5,0]),tol = 1e-8))

#print(Steepest_Descent(func = lambda x: 2*x[0]**2-1.05*x[0]**4+x[0]**6/6+x[0]*x[1]+x[1]**2,x0=np.array([10,0]),tol = 1e-8))
    
#print(Steepest_Descent(func = lambda x: x[0]**2+x[1]**2+x[2]**2,x0=np.array([10,10,10]),tol = 1e-8))

#print(Steepest_Descent(func = lambda x,y:(x-2*y**2)**2+(5*y-20)**2,x0=np.array([0.5,0]),tol = 1e-8))


# Resultados

# Sem a aplicação do momento - tol = 1e-8: f(x*) = 5.290000002091694e-14 -   número de iterações: 461
# Com a aplicação do momento - tol = 1e-8,bk = 1e-8: f(x*) = 1.4400000011486304e-14 número de iterações: 225
# Com a aplicação do momento - tol = 1e-8,bk = 1e-1: f(x*) = 3.999999995789153e-16 número de iterações: 70
# Com a aplicação do momento - tol = 1e-8,bk = 1e-3: f(x*) = 7.839999997963989e-14 número de iterações: 393
# Com a aplicação do momento Nesterov - tol = 1e-8,bk = 1e-3: f(x*) = 1.469000000796137e-13 número de iterações: 162
# Com a aplicação do momento - tol = 1e-8,bk = 1e-5: f(x*) = 5.290000002091694e-14 número de iterações: 287
# Com a aplicação do momento de Nesterov - tol = 1e-8,bk = 1e-5: f(x*) = 4.900000002613274e-15 número de iterações: 188

