# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:58:02 2019

@author: Luan
"""
import numpy as np
from scipy import optimize
import Golden_Search as gs
import bracket_search as bs
from strong_backtracking import strong_backtracking as sb
import BLS as bls
from ghpkg import gradient as gradient
from linesearch import linesearch as ls
from profiler import profile
#file = open('road.txt', 'w') 


#import Golden_Search as gs

@profile
def Steepest_Descent(func,x0,tol):
    #file.writelines(str(x0[0])+'\t'+str(x0[1])+'\n')
    dk = gradient(func,x0,1e-5)
    if np.linalg.norm(dk,2)!=0: 
        dk = dk/np.linalg.norm(dk,2)
    else: return x0
    i = 1
    #alpha_k = gs.Golden_Search(lambda a: func(x0[0]-dk[0]*a,x0[1]-dk[1]*a),0,2,1e-8) # algoritmo de razão áurea
    interval = bs.bracket_minimum(lambda a: func(x0-dk*a))
    alpha_k = optimize.minimize_scalar(lambda a: func(x0-dk*a),bounds = [interval[0],interval[1]],method = 'brent')
    #alpha_k = bls.backtracking_search(func,dk,g=dk,xk =x0,ak =1) # busca intervalar
    #alpha_k = sb(func,x0,d=-dk)
    xk = x0-alpha_k.x*dk
    #xk = x0-alpha_k*dk
    #xk_ant = xk
    while np.linalg.norm(dk*alpha_k.x,2)>tol:
    #while np.linalg.norm(dk*alpha_k,2)>tol:
        i +=1
        dk = gradient(func,xk,1e-5)
        if np.linalg.norm(dk,2)!=0: 
            dk = dk/np.linalg.norm(dk,2)
        else: return xk
        #alpha_k = gs.Golden_Search(lambda a: func(xk_ant[0]-dk[0]*a,xk_ant[1]-dk[1]*a),0,2,1e-8)
        interval = bs.bracket_minimum(lambda a: func(xk-dk*a))
        alpha_k = optimize.minimize_scalar(lambda a: func(xk-dk*a),bounds = [interval[0],interval[1]],method = 'brent')
        #alpha_k = bls.backtracking_search(func,dk,g=dk,xk = xk,ak = alpha_k)
        #alpha_k = sb(func,xk,d = -dk)
        xk = xk-alpha_k.x*dk
        #xk = xk-alpha_k*dk
        #xk_ant = xk
        #file.writelines(str(xk[0])+'\t'+str(xk[1])+'\n')
        #print(xk, 'número de iterações: {}'.format(i))
    return xk


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



 
    
    
    
    
