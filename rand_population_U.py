# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:06:26 2019

@author: Luan
"""
#import matplotlib.pyplot as plt

import numpy as np

def rand_population_U(m,a,b):
    d = len(a)
    return np.array([a+np.random.sample(d)*(b-a) for i in range(m)])

# teste
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')  
#a = np.array([1,1,1])
#b = np.array([2,3,3])
#m = 10000
#A = (rand_population_U(m,a,b))
#xs,ys,zs = A[:,0],A[:,1],A[:,2]
#
#ax.scatter(xs, ys, zs)
#ax.set_xlabel('X1')
#ax.set_ylabel('X2')
#ax.set_zlabel('X3')
