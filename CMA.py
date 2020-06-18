# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:43:00 2019

@author: Luan
"""

# Covariance Matrix Adaptation

import numpy as np
from scipy.linalg import fractional_matrix_power #scipy.linalg.fractional_matrix_power
from scipy.stats import multivariate_normal

def covariance_matrix_adaptation(f,x,k_max,sigma = 1.0):
    m = 4+np.floor(3*np.log(len(x)))
    m_elite = m//2
    mu,n = x,len(x)
    ws = np.concatenate(([np.log((m+1)/2)-np.log(i+1) for i in range(int(m_elite))],np.zeros(int(m-m_elite))))
    ws = ws/np.linalg.norm(ws)
    mu_eff = 1/sum(ws**2) 
    cs = (mu_eff + 2)/(n+mu_eff + 5)
    dsig = 1 + 2*max(0, np.sqrt((mu_eff - 1)/(n + 1)) - 1) + cs
    cS = (4 + mu_eff/n)/(n + 4 + 2*mu_eff/n)
    c1 = 2/((n + 1.3)**2 + mu_eff)
    c_mu = min(1-c1,2*(mu_eff - 2 + 1/mu_eff)/((n + 2)**2 + mu_eff))
    E = n**0.5*(1 - 1/(4*n) + 1/(21*n**2))
    ps, pS, S = np.zeros(n), np.zeros(n),np.eye(n)
    for k in range(k_max):
        P = multivariate_normal(mu,sigma**2*S)
        xs = P.rvs(int(m))
        Is = np.argsort([f(x) for x in xs])
        # seleção e atualização da média
        
        ds = [(x-mu)/sigma for x in xs]
        dw = sum([ws[i]*ds[Is[i]] for i in range(int(m_elite))])
        mu +=sigma*dw
        
        # controle do passo
        
        C = fractional_matrix_power(S,-0.5)
        ps = (1-cs)*ps+np.sqrt(cs*(2-cs)*mu_eff)*C.dot(dw)
        sigma *= np.exp(cs/dsig * (np.linalg.norm(ps)/E - 1))
        # adaptação da covariância
        
        hs = int(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(k+1))) < (1.4+2/(n+1))*E)
        pS = (1-cS)*pS + hs*np.sqrt(cS*(2-cS)*mu_eff)*dw
        w0 = [ws[i] if ws[i]>=0 else n*ws[i]/np.linalg.norm(C.dot(ds[i]))**2 for i in range(int(m))]
        pS,ds = np.matrix(pS),np.matrix(ds)
        S = (1-c1-c_mu)*S+c1*(pS.T.dot(pS)+(1-hs)*cS*(2-cS)*S)+c_mu*sum(w0[i]*(ds[Is[i]]).T.dot(ds[Is[i]])for i in range(int(m)))
        S = np.triu(S)+np.triu(S,1).T
        #print(S)
        #print(mu)
    return (mu)


#f = lambda x:x[0]**2+x[1]**2
f = lambda x: np.sin(x[1])*np.exp((1-np.cos(x[0]))**2)+np.cos(x[0])*np.exp((1-np.sin(x[1]))**2)+(x[0]-x[1])**2#(x[1]-5.1/(4*np.pi**2)*x[0]**2+5/np.pi*x[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[0])+10#x[0]**2+x[1]**2
x = np.array([1.0,1.0])
k_max = 400
print(covariance_matrix_adaptation(f,x,k_max))
print(f(covariance_matrix_adaptation(f,x,k_max)))

