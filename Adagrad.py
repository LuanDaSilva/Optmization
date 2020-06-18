# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:56:00 2019

@author: Luan
"""

# Adagrad:

import numpy as np
from scipy import optimize
#import Golden_Search as gs
#import bracket_search as bs
#from strong_backtracking import strong_backtracking as sb
#import BLS as bls
from ghpkg import gradient as gradient
#from linesearch import linesearch as ls
#file = open('road.txt', 'w') 


#import Golden_Search as gs


def Adagrad(func,x0,tol):
    #file.writelines(str(x0[0])+'\t'+str(x0[1])+'\n')
    alpha_k = 0.01
    e = 1e-8
    #sk = 0
    aux_sk = []
    dk = gradient(func,x0,1e-5)
    if np.linalg.norm(dk,2)!=0: 
        dk = dk/np.linalg.norm(dk,2)
    else: return x0
    aux_sk.append(dk)
    xk = x0-alpha_k*dk/(e+np.sqrt(sum([gi**2 for gi in aux_sk])))
    while np.linalg.norm(dk/(e+np.sqrt(sum([gi**2 for gi in aux_sk])))*alpha_k,2)>tol:
        dk = gradient(func,xk,1e-5)
        if np.linalg.norm(dk,2)!=0: 
            dk = dk/np.linalg.norm(dk,2)
        else: return xk
        aux_sk.append(dk)
        xk = xk-alpha_k*dk/(e+np.sqrt(sum([gi**2 for gi in aux_sk])))
        print(xk)
    return xk

print(Adagrad(func = lambda x: x[0]**2+x[1]**2,x0=np.array([1.5,1.5]),tol = 1e-4))
#print(Steepest_Descent(func = lambda x:(x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2,x0=np.array([1,1]),tol = 1e-8))
#print(Steepest_Descent(func = lambda x:2*x[0]**2-1.05*x[0]**4+x[0]**6/6+x[0]*x[1]+x[1]**2,x0=np.array([0.01,1]),tol = 1e-8))
#print(Steepest_Descent(func = lambda x:(x[0]**2)+(x[1]**2),x0=np.array([1,1]),tol = 1e-8))
#print(Steepest_Descent(func = lambda x,y:(x-2)**2+(y-4)**2,x0=np.array([100,100]),tol = 1e-8))
#print(Steepest_Descent(func = lambda x: (1-x[0])**2+100*(x[1]-x[0]**2)**2,x0=np.array([10,10]),tol = 1e-8))
#file.close()



# Para a função: (x-2*y**2)**2+(5*y-20)**2

# backing tracking line search:  7.487462841527536e-11
# backing tracking line search(normalizando): 2.8183311938764397e-08
# strong backing tracking line search: 1.3948102349780751e-11
# strong backing tracking line search (normalizado) :  1.164999956529611e-12
# busca otimizada:  1.0321000504267392e-12
# busca otimizada(normalizada): 4.00000009348878e-14
# busca "dourada" (normalizada): 2.3869001349984297e-12
# busca "dourada": 1.4834902345538928e-11