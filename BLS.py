# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:40:22 2019

@author: Luan
"""

# Controle do alpha

def backtracking_search(func,dk,g,xk,ak,p=0.5,b = 1e-4):
    while func(xk-ak*dk)> func(xk)-b*ak*g.dot(dk):
        ak*=p
    return ak



