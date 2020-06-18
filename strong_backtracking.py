# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:15:17 2019

@author: Luan
"""
from ghpkg import gradient as grad
#from gradient import gradient as grad
def strong_backtracking(func,x,d,a = 1,b= 1e-4,sigma = 0.1):
    k = 0
    #print('yay!')
    y0,g0,y_prev,a_prev = func(x),grad(func,x,1e-4).dot(d),None,0
    al0,ahi = None,None
    
    # redução do intervalo
    
    while True:
        #print('yay 2!')
        y = func(x+a*d)
        if y > y0 + b*a*g0 or (y_prev != None and y >= y_prev):
            al0,ahi = a_prev,a
            break 
        g = grad(func,x+a*d,1e-4).dot(d)
        if abs(g)<=-sigma*g0:
            return a
        elif g>=0:
            al0,ahi = a,a_prev
            break
        y_prev,a_prev,a = y,a,2*a
    ylo = func(x+al0*d)
    while True:
        #print('yay 3!')
        a = (al0+ahi)/2
        y = func(x+a*d)
        if y > y0+b*a*g0 or y>=ylo:
            ahi = a
        else:
            g = grad(func,x+a*d,1e-4).dot(d)
            if abs(g) <=-sigma*g0:
                return a
            elif g*(ahi-al0)>=0:
                ahi = al0
            al0 = a
        k +=1
        if k == 100: 
            return a
            break
    
# teste
            
