# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 20:39:27 2019

@author: Luan
"""
import numpy as np
#import Golden_Search as gs
from scipy import optimize
from profiler import profile
import bracket_search as bs
from ghpkg import gradient as gradient
from ghpkg import Hessian as Hessian
#def Hessian(func,x0,h):
#    H = np.zeros([len(x0),len(x0)])
#    H11 = (func(x0[0]+h,x0[1])+func(x0[0]-h,x0[1])-2*func(x0[0],x0[1]))/(h**2)
#    H22 = (func(x0[0],x0[1]+h)+func(x0[0],x0[1]-h)-2*func(x0[0],x0[1]))/(h**2)
#    H12 = (func(x0[0]+h,x0[1]+h)+func(x0[0]-h,x0[1]-h)-func(x0[0]+h,x0[1]-h)-func(x0[0]-h,x0[1]+h))/(4*h**2)
#    H[0][0] = H11
#    H[1][1] = H22
#    H[0][1] = H12
#    H[1][0] = H12
#    return H
#def gradient(func,x0,h):
#    grad = []
#    grad.append((func(x0[0]+h,x0[1])-func(x0[0]-h,x0[1]))/(2*h))
#    grad.append((func(x0[0],x0[1]+h)-func(x0[0],x0[1]-h))/(2*h))
#    return np.array(grad)
def positive_def(H):
    return np.all(np.linalg.eigvals(H) > 0)
@profile
def Newton_Method(func,x0,tol):
    n_iter = 500
    i = 0
    beta = 1e4
    g = gradient(func,x0,h=1e-4)
    H = Hessian(func,x0,h=1e-4)
    print(H)
    if not positive_def(H):
        H = (H+beta*np.eye(2,2))/(1+beta)
    dk = np.linalg.inv(H).dot(g)
    dk = dk/np.linalg.norm(dk,2)
    #alpha_k = gs.Golden_Search(lambda a: func(x0[0]-dk[0]*a,x0[1]-dk[1]*a),0,1,1e-8)
    interval = bs.bracket_minimum(lambda a: func(x0-dk*a))
    alpha_k = optimize.minimize_scalar(lambda a: func(x0-dk*a),bounds = [interval[0],interval[1]],method = 'brent')
    xk = x0-alpha_k.x*dk
    while np.linalg.norm(dk*alpha_k.x,2) and i<n_iter:
        g = gradient(func,xk,h=1e-4)
        H = Hessian(func,xk,h=1e-4)
        if not positive_def(H):
            H = (H+beta*np.eye(2,2))/(1+beta)
        dk = np.linalg.inv(H).dot(g)
        dk = dk/np.linalg.norm(dk,2)
        interval = bs.bracket_minimum(lambda a: func(xk-dk*a))
        alpha_k = optimize.minimize_scalar(lambda a: func(xk-dk*a),bounds = [interval[0],interval[1]],method = 'brent')
        #alpha_k = gs.Golden_Search(lambda a: func(xk[0]-dk[0]*a,xk[1]-dk[1]*a),0,1,1e-8)
        xk = xk-alpha_k.x*dk
        i+=1
        #print(xk,'número de iterações {}'.format(i))
    return xk

#print(Newton_Method(func = lambda x,y:(x-2*y**2)**2+(5*y-20)**2,x0=np.array([0,0]),tol = 1e-8))
print(Newton_Method(func = lambda x:100*(1-x[0])**2+(x[1]-x[0]**2)**2,x0=np.array([0,0]),tol = 1e-8))
#print(Newton_Method(func = lambda x,y:(x+2*y-7)**2+(2*x+y-5)**2,x0=np.array([0.5,0]),tol = 1e-8))

# busca otimizada: 2.900000124622787e-15
# busca otimizada(normalizada): 2.900000124622787e-15


    
    
