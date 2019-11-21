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
np.random.seed(3)


def f(X, n, r1, r2):
    alpha= X[0]
    beta = X[1]
    t1 = (beta**(n*alpha))*(r1**(alpha-1.0))*(np.exp(  -beta*(r2+1.0) ))
    t2 = gg(alpha)**n
    return t1/t2
    
    
def Metropolis_Hastings(maxite, cov, mu, sigma, burnin):
    delta = 1.0
    #x_t = np.array([uniform.rvs(mu[0]-delta, mu[0]+delta), uniform.rvs(mu[1]-delta, mu[1]+delta)]) # en el soporte...
    x_t =np.array([1000,1])# np.array([uniform.rvs(mu[0]-delta, mu[0]+delta), uniform.rvs(mu[1]-delta, mu[1]+delta)]) # en el soporte...
    X_walk =  np.copy(x_t)
    cov_prop = np.identity(2)*sigma
    fx_t = multivariate_normal.pdf(x_t, mu, cov)
    print(cov_prop)
    cont = 0
    while cont < maxite:
     y_t = multivariate_normal.rvs(x_t, cov_prop)
     if y_t[0] <= 0 or y_t[1] <=0:
        continue
     fy_t = multivariate_normal.pdf(y_t, mu, cov)
     rho = min(1.0, fy_t/fx_t)
     if uniform.rvs(0.0, 1.0) <= rho:
       if cont > burnin:
        X_walk = np.vstack((X_walk, y_t))
       x_t = np.copy(y_t)
       fx_t = fy_t
       cont +=1
       print(cont)

    return X_walk
maxite = 80000
cov = np.array([[1.0, 0.9],[0.9, 1.0]])
mu = np.array([3.0, 5.0])
sigma = 10.9
burnin=1000

X = Metropolis_Hastings(maxite, cov, mu, sigma, burnin)
#plt.plot(X, FX, '.')
#plt.show()

dist = 5.0
xl, yl = np.mgrid[(mu[0]-dist):(mu[0]+dist):.1, (mu[1]-dist):(mu[1]+dist):.1]
zl = np.copy(xl)
for i in range(0, len(xl[0,:])):
  for j in range(0, len(xl[0,:])):	
     zl[i,j] = multivariate_normal.pdf(np.array([xl[i,j], yl[i,j]]), mu, cov)

plt.contourf(xl, yl, zl)
plt.plot(X[:,0], X[:,1])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")

plt.show()

