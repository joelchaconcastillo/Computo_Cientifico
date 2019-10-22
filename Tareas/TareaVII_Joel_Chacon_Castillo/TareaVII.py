import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm
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


def f(p, n, r):
    fp =(p**r)*( (1.0-p)**(n-r))*np.cos(np.pi*p)
    if fp >=0 and fp <= 0.5:
      return fp
    else:
      return 0.0
def posterior(r1, r2, alpha, beta, n):
   
    prod1 = beta**(n*alpha)/(    )
    prod2 = r1**(alpha-1.0) 
    prod3 = np.exp(-beta*(r2+1.0))
    return prod1*prod2*prod3
     
def Metropolis_Hastings(n, maxite):
    alpha = 3
    beta = 100
    ###Generating X_i with Ga(alpha, beta)
    X =  gamma.rvs(alpha, beta, size=(n))
    r2 = np.sum(X)
    r1 = np.prod(X)
     
    print(X)
    exit(0) 
#
#    X = np.array([])
#    FX = np.array([])
#    a_beta = r+1.0
#    b_beta = n-r+1.0
#    ##given X^(t)
#    x_t =  uniform.rvs(0.0, 0.5)  
#    X =np.append(X, x_t)
#    FX =np.append(FX, f(x_t, n, r))
#    cont = 0
#    while cont < maxite:
#       #generate 
#       y_t = beta.rvs(a_beta, b_beta)
#       fx = f(x_t, n, r)
#       fy = f(y_t, n, r)
#       if fx == 0:
#           continue;
#       cont +=1
#       ratio_f = fy/fx
#       ratio_q = beta.pdf(x_t, a_beta, b_beta)/beta.pdf(y_t, a_beta, b_beta)
#       rho = min(ratio_f*ratio_q, 1.0)
#       u_t =  uniform.rvs(0, 1.0)  
#       if u_t <= rho:
#           x_t = y_t
#           X = np.append(X, y_t)
#           FX =np.append(FX, fy)
#       else:
#           X = np.append(X, x_t)
#           FX =np.append(FX, fx)
#    plt.hist(X, label='CDF',histtype='step', alpha=0.8, color='k', density=True )
#    plt.title("Beta instrumental distribution with r: " + str(r) + ", n:" + str(n)+", p:"+str(p))
#    plt.show()
#    return X, FX


n = 3
maxite = 10000
[X, FX] = Metropolis_Hastings(n, maxite)
plt.plot(X, FX, '.')
plt.show()
