# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:32:00 2019

@author: Luan
"""
import numpy as np
from profiler import profile
def basis(i,n):
    d = np.zeros(n)
    d[i-1] = 1
    return d
def Simplex(n):
    S = [basis(i,n) for i in range(1,n+1)]
    S.append(np.zeros(n))
    return S


@profile
def nelder_mead(f,n,tol,a = 1,b = 2,c = 0.5):
    S = Simplex(n)
    err,y_arr = 10**10,[f(x) for x in S]
    while err>tol:
#        print('entrei 2!')
        #print(S[np.argmin(y_arr)])
        S = [S[i] for i in np.argsort(y_arr)]
        y_arr.sort()
        xl,yl = S[0],y_arr[0]
        xh,yh = S[len(y_arr)-1],y_arr[len(y_arr)-1]
        xs,ys = S[len(y_arr)-2],y_arr[len(y_arr)-2]
        xm = np.mean(S[0:len(S)-1],axis=0)
        #print(xm)
        #break
        xr = xm+a*(xm-xh)
        yr = f(xr)
        if yr<yl:
            xe = xm+b*(xr-xm)
            ye = f(xe)
            if ye<yr:
                S[len(y_arr)-1],y_arr[len(y_arr)-1] = xe,ye
            else:
                 S[len(y_arr)-1],y_arr[len(y_arr)-1] = xr,yr
        elif yr>ys:
            if yr<=yh:
                xh,yh,S[len(y_arr)-1],y_arr[len(y_arr)-1] = xr,yr,xr,yr
            xc = xm+c*(xh-xm)
            yc = f(xc)
            if yc>yh:
                for i in range(1,len(y_arr)-1):
                    S[i] = (S[i]+xl)/2
                    y_arr[i] = f(S[i][0],S[i][1])
            else:
                 S[len(y_arr)-1],y_arr[len(y_arr)-1] = xc,yc
        else:
            S[len(y_arr)-1],y_arr[len(y_arr)-1] = xr,yr
        err = np.std(y_arr)
        #print(S[np.argmin(y_arr)])
    return S[np.argmin(y_arr)]

#f = lambda x:(x[1]-5.1/(4*np.pi**2)*x[0]**2+5/np.pi*x[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[0])+10
#print(nelder_mead(lambda x: (1-x[0])**2+(x[1]-x[0]**2)**2+x[2]**2,3,1e-12))
#print(nelder_mead(f,2,1e-12))
            
        
                 
                
                    
            

        
        