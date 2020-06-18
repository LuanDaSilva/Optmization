# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:29:32 2019

@author: Luan
"""

def bracket_minimum(f, x=0, s=1e-2, k=2.0):
    a,ya = x,f(x) 
    b, yb = a + s, f(a + s)
    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    while True:
        c, yc = b + s, f(b + s)
        if yc > yb:
            if c<a:
                return c,a
            else:
                return a,c
        a, ya, b, yb = b, yb, c, yc
        s *= k
        
#print(bracket_minimum(lambda x: x**5-10*x**3-89))


        
