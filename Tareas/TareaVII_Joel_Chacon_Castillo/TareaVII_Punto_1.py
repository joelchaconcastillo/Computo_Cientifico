import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm, expon, multivariate_normal
from scipy.special import gamma as gg, factorial
from scipy import stats
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import sys
np.set_printoptions(threshold=sys.maxsize)
#np.random.seed(3)


def f(X, n, r1, r2):
    alpha= X[0]
    beta = X[1]
    t1 = (beta**(n*alpha))*(r1**(alpha-1.0))*(np.exp(  -beta*(r2+1.0) ))
    t2 = gg(alpha)**n
    return t1/t2
    
    
def Metropolis_Hastings(n, maxite, sigma1, sigma2):
    alpha = 3
    beta = 100
    X =  gamma.rvs(alpha, 1.0/beta, size=(n))
    r2 = np.sum(X)
    r1 = np.prod(X)
    x_t = np.array([uniform.rvs(1.0, 4.0), uniform.rvs(1.0, 2.0)]) ###alpha, beta..
    X_walk =  np.copy(x_t)
    cov = np.array([[sigma1**2, 0],[0, sigma2**2]])
    cont = 0
    while cont < maxite:
     y_t = multivariate_normal.rvs(x_t, cov)
     if y_t[0] < 1.0 or y_t[0] > 4.0:
        continue
     if y_t[1] <= 1.0:
        continue

     fx_t = f(x_t, n, r1, r2) 
     fy_t = f(y_t, n, r1, r2)
     print(fx_t)
#     print(fy_t)
#     print(fy_t/fx_t)
#     if fy_t <1e-100:
#       continue
     print(cont)   
     rho = min(1.0, fy_t/fx_t)
     if rho < uniform.rvs(0.0, 0.5):
        X_walk = np.vstack((X_walk, y_t))
        x_t = np.copy(y_t)
        cont +=1

    return X_walk
n = 30
maxite = 1000
sigma1 = 0.1
sigma2 = 0.01
[X, FX] = Metropolis_Hastings(n, maxite, sigma1, sigma2)
plt.plot(X, FX, '.')
plt.show()
