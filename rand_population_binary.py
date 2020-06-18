# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:36:09 2019

@author: Luan
"""
import numpy as np

bitrand = lambda m,n:[np.random.randint(2, size=m) for i in range(n)]
bitrand(1000,2)