# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:58:27 2019

@author: Luan
"""
import cProfile, pstats, io


def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile(timeunit=0.000001)
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        #print(s.getvalue())
        
        file = open("log.txt", 'w')
        file2 = open('{}_log.txt'.format(fnc.__name__),'w')
        file.write(s.getvalue())
        file.close()
        file = open('log.txt','r')
        lines = file.readlines()
        file.close()
        for i in range(3,len(lines)):
            file2.write(','.join(lines[i].lstrip().split()[0:6])+''.join(lines[i].lstrip().split()[6:])+'\n')       
        file2.close()
        return retval
    return inner

    
# para criar o data frame:
#df =  pd.read_csv('nome do arquivo.txt',encoding = 'latin')