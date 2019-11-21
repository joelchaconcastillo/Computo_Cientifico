import time
import pandas as pd
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
    
    
def Metropolis_Hastings(maxite, alpha, beta, burnin):
    
    delta = 1.0
    mu = alpha*beta
    x_t = np.array([uniform.rvs(mu-delta, mu+delta)]) # en el soporte...
    X_walk =  np.copy(x_t)
    fx_t = gamma.pdf(x_t, alpha, 1.0/beta) 
    qx_t = gamma.pdf(x_t, int(alpha), 1.0/beta) 
    cont = 0
    while cont < maxite:
     y_t = gamma.rvs(int(alpha), 1.0/beta)
     fy_t = gamma.pdf(y_t, alpha, 1.0/beta)
     qy_t = gamma.pdf(y_t, int(alpha), 1.0/beta)
     rho = min(1.0, (fy_t/fx_t)*(qx_t/qy_t))
     if uniform.rvs(0.0, 1.0) <= rho:
       if cont > burnin:
        X_walk = np.vstack((X_walk, y_t))
       x_t = np.copy(y_t)
       fx_t = fy_t
       qx_t = qy_t
       cont +=1
       print(cont)

    return X_walk
maxite = 10000
alpha=1.3
beta=1
burnin=100

X = Metropolis_Hastings(maxite, alpha, beta, burnin)
FX = gamma.pdf(X, alpha, 1.0/beta)


p =  pd.DataFrame(X).hist(bins=100, range=(0,5), figsize=(9,9))
plt.title("Estimacion del pdf de una distribucion Gamma(" + str(alpha)+","+str(beta)+") \n con una propuesta  Gamma(" + str(int(alpha))+","+str(beta)+")")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
plt.plot(X, FX, '.')
plt.title("Valor de la distribución Gamma(" + str(alpha)+","+str(beta)+")")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

#plt.plot(X[:,0], X[:,1])
#plt.xlabel(r"$\alpha$")
#plt.ylabel(r"$\beta$")

#plt.show()

