import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm, expon, multivariate_normal, lognorm, cauchy, norm
from scipy.special import gamma as gg, factorial
import random
from numpy.random import choice

from scipy import stats
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(3)

   
    
def Metropolis_Hastings(maxite, cov, mu, sigma1, sigma2, burnin, batch):
    delta = 1.0
    x_t = np.array([uniform.rvs(0.0, 5.0), uniform.rvs(0.0, 5.0)]) # en el soporte...
    #x_t =np.array([1000,1])# np.array([uniform.rvs(mu[0]-delta, mu[0]+delta), uniform.rvs(mu[1]-delta, mu[1]+delta)]) # en el soporte...
    X_walk =  np.copy(x_t)
    cov_prop1 = np.identity(2)*sigma1
    cov_prop2 = np.identity(2)*sigma2
    fx_t = multivariate_normal.logpdf(x_t, mu, cov)
    cont = 0.0
    w1 = 0
    w2 = 0
    pb1 = 0.5
    pb2 = 0.5
    total = 0.0
    PesosH = np.array([0.5, 0.5])
    while cont < maxite:
     total +=1
     idx = choice([0,1],p= [pb1, pb2])
     pb =  uniform.rvs(0.0, 1.0)
     if idx == 0:
       y_t = multivariate_normal.rvs(x_t, cov_prop1)
     if idx == 1:
       y_t = multivariate_normal.rvs(x_t, cov_prop2)
     if y_t[0] <= 0 or y_t[1] <=0:
        continue
     fy_t = multivariate_normal.logpdf(y_t, mu, cov)
     rho = fy_t-fx_t
     if np.log(uniform.rvs(0.0, 1.0)) <= rho:
       if idx ==0:
          w1 += 1
       if idx ==1:
          w2 += 1
       if cont > burnin:
        X_walk = np.vstack((X_walk, y_t))
       x_t = np.copy(y_t)
       fx_t = fy_t
       cont +=1
       if (cont % batch ) == 0:
         pb1 = w1/(w1+w2)
         pb2 = w2/(w1+w2)
         PesosH = np.vstack((PesosH, np.array([pb1, pb2])))
         w1 = 0.0
         w2 = 0.0
    print("Eficiencia " + str(cont/total))
    pesos = np.array([pb1, pb2])
    return X_walk[1:,:], cont/total, PesosH
def MH(maxite, d, Cov):
  x_t = [  uniform.rvs(0.0, 0.4) for i in range(d)  ]
  log_fx_t =  multivariate_normal.logpdf(x_t, np.ones(d), Cov)
  n = 0
  beta = 0.05
  X = np.copy(x_t)
  X2 = np.copy(x_t)
  mix_cov = (0.1**2)*np.ones(d)/(float(d))
  empirical_cov = np.zeros((d,d));
  sum_cov = np.zeros((d,d))
  t = 0.0
  meanx = np.zeros(d)
  while n < maxite:
     t += 1.0
     if n <= 2*d:
        y_t = multivariate_normal.rvs(x_t, mix_cov)
     else:
        idx = choice([0,1],p= [1.0-beta, beta])
        if idx == 0:
           Covp = np.cov(X2.T)
           Covp2 = (2.381204**2)*Covp/(float(d)**0.5)
           y_t = multivariate_normal.rvs(x_t, Covp2)
        elif idx ==1:
           y_t = multivariate_normal.rvs(x_t, mix_cov)
     log_fy_t = multivariate_normal.logpdf(y_t, np.zeros(d), Cov)
     if np.log(uniform.rvs(0.0, 1.0)) < log_fy_t-log_fx_t:
       X = np.vstack((X, y_t))
       x_t = y_t
       fx_t = log_fy_t
       n+=1
     X2 = np.vstack((X2, x_t))
     ##Incremental mean and incremental variance....
     oldmeanx = np.copy(meanx)
     meanx = (t/(t+1.0))*oldmeanx + (1.0/(t+1.0))*x_t
  return X
    
maxite = 1000
d = 100 #dimension...
M = np.zeros((d, d))
for i in range(d):
  for j in range(d):
    M[i,j] = norm.rvs(0.0, 1.0)
Cov = M.dot(M.T)
print(Cov)
MH(maxite, d, Cov) 
#X, eficiencia, PesosH = Metropolis_Hastings(maxite, cov, mu, sigma1, sigma2, burnin, batch)


