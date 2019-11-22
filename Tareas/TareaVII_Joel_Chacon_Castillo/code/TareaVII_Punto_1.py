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
np.random.seed(30)


def f(X, n, r1, r2):
    alpha= X[0]
    beta = X[1]
    t1 = beta**(n*alpha)
    t2 = r1**(alpha-1.0)
    t3 = np.exp(-beta*(r2+1.0))
    t4 = 1.0/(gg(alpha)**n)
    return t1*t2*t3*t4
def fboth(X, Y, n, r1, r2):
    alpha1= X[0]
    beta1 = X[1]
    alpha2= Y[0]
    beta2 = Y[1]
    t1 = ((beta1**(alpha1))/(beta2**(alpha2)))**n
    t2 = (r1**(alpha1-1.0))/(r1**(alpha2-1.0))
    t3 = np.exp(-(beta1*(r2+1.0)-beta2*(r2+1.0)))
    t4 = 1.0/(gg(alpha1)/gg(alpha1))**n
    return t1*t2*t3*t4

    
    
def Metropolis_Hastings(n, maxite, sigma1, sigma2, r1, r2, burnin):
    x_t = np.array([uniform.rvs(1.0, 4.0), uniform.rvs(0.0, 4.0)]) ###alpha, beta..
    X_walk =  np.copy(x_t)
    cov = np.array([[sigma1**2, 0],[0, sigma2**2]])
    cont = 0
    while cont < maxite:
     y_t = multivariate_normal.rvs(x_t, cov)
     if y_t[0] < 1.0 or y_t[0] > 4.0:
        continue
     if y_t[1] <= 0.0 or y_t[1] > 5.0:
        continue
     fx_t = f(x_t, n, r1, r2) 
     fy_t = f(y_t, n, r1, r2)
     rho = min(1.0, fy_t/fx_t)

#     fy_x_t = fboth(y_t, x_t, n, r1, r2) 
#     rho = min(1.0, fy_x_t)


     if uniform.rvs(0.0, 1.0) <= rho:
       if burnin < cont:
        X_walk = np.vstack((X_walk, y_t))
       x_t = np.copy(y_t)
       cont +=1

    return X_walk[1:,:]
n = 30
sigma1 = 0.01
sigma2 = 0.01
alpha = 3
beta = 100
maxite = 10000
burnin = 1
X =  gamma.rvs(alpha, 1.0/beta, size=(n))
r2 = np.sum(X)
r1 = np.prod(X)
#print(X)
#print(r1)
#print(r2)
#print("****")



X = Metropolis_Hastings(n, maxite, sigma1, sigma2, r1, r2, burnin)
#l = len(X[:,1])

#plt.plot(np.abs(X[0:(l-1),0]-X[1:l,0]))
#plt.title("Desplazamientos de "r"$\alpha$")
#plt.xlabel("Iteracion")
#plt.ylabel("Deplazamiento")
#
#plt.show()
#
#plt.plot(np.abs(X[0:(l-1),1]-X[1:l,1]))
#plt.title("Desplazamientos de "r"$\beta$")
#plt.xlabel("Iteracion")
#plt.ylabel("Deplazamiento")
#plt.show()

xl, yl = np.mgrid[0.0:4.0:.1, 0.0:4.0:.1]
zl = np.copy(xl)
for i in range(0, len(xl[0,:])):
  for j in range(0, len(yl[0,:])):
     zl[i,j] = f(np.array([xl[i,j], yl[i,j]]),n, r1, r2)

plt.contourf(xl, yl, zl)
plt.plot(X[:,0], X[:,1], 'r')
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")

plt.show()

