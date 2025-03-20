# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:07:55 2025

@author: baolq
"""

import numpy as np
import math
import matplotlib.pyplot as plt


T = 4000
N = 10
t = np.arange(T+1)

max_gamma = 0.999
delta = 0.008
initial_pts = np.random.uniform(size=(10,2))*5 - 2.5

f = np.cos((t / T + delta) / (1 + delta) * np.pi / 2) ** 2
alpha = np.clip(f[1:] / f[:-1], 1 - max_gamma, 1)   
alpha = np.append(1, alpha).astype(np.float32)  # ajouter α0 = 1    
gamma = 1 - alpha    
alpha_t = np.cumprod(alpha)    

#%%

def norm_pdf_multivariate(x, mu, sigma):
    # print("x=",x)
    x,mu,sigma = np.array(x), np.array(mu), np.array(sigma)
    size = len(x)
    print(size, len(mu),sigma.shape )
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def score(phi,t):
    alpha_curr= alpha_t[int(t)]
    probs = []
    for pt in initial_pts:
        # print("phi ",phi)
        
        probs+=[norm_pdf_multivariate(phi,np.sqrt(alpha_curr)*pt,(1-alpha_curr)*np.identity(2))]
    
    probs = np.array(probs)
    
    return -(1/(1-alpha_curr))*np.sum((np.sqrt(alpha_curr)*initial_pts-phi)*probs.reshape(-1,1),axis=0)/np.sum(probs)

phi = [0,0]

phi_iter = []
phi_iter+=[phi]
for i in reversed(range(1,T+1)):
    # print("i=",i)
    s = score(phi_iter[-1],i)
    # print("phi_iter = ",phi_iter[-1])
    # print("s= ",s)
    phi_iter += [phi_iter[-1]-gamma[i]*(phi_iter[-1]+s)]
    
#%%
phi_iter = np.array(phi_iter)
coord1 = phi_iter[:,0]
coord2 = phi_iter[:,1]

plt.scatter(coord1,coord2)
plt.scatter(initial_pts[:,0],initial_pts[:,1])
plt.show()


