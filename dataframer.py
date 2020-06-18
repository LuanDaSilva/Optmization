# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:10:24 2019

@author: Luan
"""
import pandas as pd
def dataframer(file):
    A = open(file,'r')
    lines = A.readlines()
    lines[0] = '    '.join(lines[0].split())
    A.close()
    A = open(file,'w')
    for line in lines:    
        A.write(line)
    A.close()
    df = pd.read_csv(file,sep ='    ' ,encoding='latin')
    return df