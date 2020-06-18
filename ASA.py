# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:53:25 2019

@author: Luan
"""
import numpy as np
from profiler import profile


def basis(i,n):
    d = np.zeros(n)
    d[i-1] = 1
    return d

def corana_update(v,a,c,ns):
    #print('Entrada: v = {}'.format(v))
    for i in range(len(v)):
        ai,ci = a[i],c[i]
        if ai>0.6*ns:
            v[i]*= (1+ci*(ai/ns - 0.6)/0.4)
        elif ai<0.4*ns:
            v[i] /= (1+ci*(0.4-ai/ns)/0.4)
    #print('Saída: v = {}'.format(v))
    return v
@profile
def ASA(f,x,v,t,tol,ns=20,ne=4,gamma = 0.85):
    nt = max(100,5*len(x))
    c = np.empty(len(x))
    c.fill(2)
    y = f(x)
    x_best,y_best = x,y
    y_arr,n = [],len(x)
    a,counts_cycles,counts_resets = np.zeros(n),0,0
    while True:
        for i in range(n):
#            x_new = x+basis(i,n)*np.random.uniform(-1,1)*v[i] 
            if sum(v)!=0:
                x_new = x+basis(i,n)*np.random.uniform(-1,1)*v[i]
            else:
                x_new = x+basis(i,n)*np.random.uniform(-1,1)
            y_new = f(x_new)
            dy = y_new-y
            if dy<=0 or np.random.rand()<np.exp(-dy/t):
                x,y = x_new,y_new
                a[i]+=1
                if y_new<y_best:
                    x_best,y_best = x_new,y_new
        counts_cycles +=1
        if  not counts_cycles>=ns:
            continue
        counts_cycles =0
#        if sum(v)!=0: corana_update(v,a,c,ns)
        corana_update(v,a,c,ns)
        a.fill(0)
        
        counts_resets+=1
        if not counts_resets>=nt:
            continue
        counts_resets = 0
        t*=gamma
        y_arr.append(y)
        if not (len(y_arr)>ne and y_arr[-1]-y_best<=tol and all([abs(y_arr[-1]-y_arr[-1-i])<=tol for i in range(1,ne)])):
            x,y = x_best,y_best
        else:
            break
        #print(x_best)
    return x_best
        

#f = lambda x: (x[0]-4)**2+(x[1]**2+5)**2+(x[2]**3-5)**2
f = lambda x:(x[1]-5.1/(4*np.pi**2)*x[0]**2+5/np.pi*x[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[0])+10
x = np.array([0,0])
v = np.array([0.1,0.1])
t = 1
tol = 1e-8
print(ASA(f,x,v,t,tol))       
    
        
    
    
    