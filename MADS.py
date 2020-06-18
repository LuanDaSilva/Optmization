# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:37:43 2019

@author: Luan
"""
import numpy as np

def rand_positive_spanning_set(a,n):
    d = round(1/np.sqrt(a))
    L = d*np.random.choice([-1,1])*np.eye(n)
    for i in range(0,n):
        for j in range(i+1,n):
            if -d+1==0 or d-1==0:
                L[i][j] = 0
            else:
                L[i][j] = np.random.randint(-d+1,d-1)
    D = L[np.random.permutation(n),:]
    D = D[:,np.random.permutation(n)]
    D = np.vstack((D,-sum(D)))
    D = [D[i,:] for i in range(0,n+1)]
    return D
def MADS(f,x0,tol):
    a,y0,n = 1,f(x0),len(x0)
    while a>tol:
        improved = False
        for (i,d) in enumerate(rand_positive_spanning_set(a,n)):
            x = x0+a*d
            y = f(x)
            if y<y0:
                x0,y0,improved = x,y,True
                x = x0+3*a*d
                y = f(x)
                if y<y0:
                    x0,y0 = x,y
                break
        if improved:
            a = min(4*a,1)
        else:
            a = a/4
    return x0

print(MADS(lambda x:(1-x[0])**2+(x[1]-x[0]**2)**2+x[2]**2,np.array([2,2,2]),1e-9))