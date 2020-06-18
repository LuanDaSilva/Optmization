# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:00:37 2019

@author: Luan
"""
import numpy as np
def Matthew_Davies_Algorithm(H):
    # Step 1
    n = len(H)
    L = np.zeros([n,n])
    D = np.zeros([n,n])
    if H[0][0]>0:
        H00 = H[0][0]
    else:
        H00 = 1
    # Step 2
    for k in range(1,n):
        m = k-1
        L[m][m] = 1
        if H[m][m]<=0:
            H[m][m] = H00
        for i in range(k,n):
            L[i][m] = -H[i][m]/H[m][m]
            H[i][m] = 0
            for j in range(k,n):
                H[i][j]+= L[i][m]*H[m][j]
        if 0<H[k][k] and H[k][k]<H00: 
            H00 = H[k][k]
        #print(H00)
    #H00 = -H[n-1][n-1]
    # Step 3
    
    L[n-1][n-1] = 1
    if H[n-1][n-1]<=0:
        H[n-1][n-1] = H00
    for i in range(n):
        D[i][i] = H[i][i]
    return(L,D)

# Teste
    
#H = np.array([[3,-6],[-6,59/5]])
#L,D = Matthew_Davies_Algorithm(H)
#print('L:\n {} '.format(L),'\nD:\n {} '.format(D),'\nH:\n {} '.format(H))
    
    
        
            