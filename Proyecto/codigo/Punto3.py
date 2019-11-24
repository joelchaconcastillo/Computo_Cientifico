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


###global params 
mu_p = 0.0
sigma_p = a1 = a2 = b1 = b2 = 1.0

def log_f1(Y, R, x):
 global mu_p, sigma_p, a1, a2, b1, b2
 d = len(x)
 A = np.exp(x[0])
 V = np.exp(x[1])
 mu = x[2]
 sum = -((mu-mu_p)**2)/(2.0*(sigma_p**2))
 sum -= b1/A - (a1+1.0)*np.log(A)
 sum -= b2/V - (a2+1.0)*np.log(V)
 for ii in range(3, d):
  sum += log_f2(Y, R, x, ii)
 return sum

def log_f2(Y, R, x, ii):
 global mu_p, sigma_p, a1, a2, b1, b2
 A = np.exp(x[0])
 V = np.exp(x[1])
 mu = x[2]
 sum = -np.log(A) - ((x[ii]-mu)/A)**2
 sum += np.sum(np.power(  Y[ii-3, 0:R[ii-3]] - x[ii], 2)/(2.0*V))
 return sum


def MH(NBatches, Batchsize, Y, R):
  d = 3 + len(R)
  x_t = np.ones(d) ## A,V, mu, theta_i
  x_t[0:3]=1.0
  log_sigma = np.zeros(d)
  X = np.copy(x_t)
  log_fx_t =  log_fy_t = 0
  sigma = 1
  n = 0
  t = 0.0
  fraction_accepted = 0.44
#  while t < maxite:
  for t in range(1, NBatches):
    count_variable= np.zeros(d)
    for batch in range(1,Batchsize):
     print(batch)
#     t += 1.0
     #propose movement
     #y_t = multivariate_normal.rvs(x_t, np.exp(np.power(log_sigma, 2)).dot(np.ones(d,d))) 
#     y_t = x_t + np.exp(log_sigma).dot(multivariate_normal.rvs(np.zeros(d), np.ones(d,d)))
     y_t = x_t + np.multiply(np.exp(log_sigma),(norm.rvs(0,1.0, size=d)))
     y_t[0] = max(0.0, y_t[0])
     y_t[1] = max(0.0, y_t[1])
     print(x_t)
     for ii in range(d):
      if ii < 3:
          log_fy_t = log_f1(Y, R, y_t)
          log_fx_t = log_f1(Y, R, x_t)
      else:
          log_fy_t = log_f2(Y, R, y_t, ii)
          log_fx_t = log_f2(Y, R, x_t, ii)
      if np.log(uniform.rvs(0.0, 1.0)) < log_fy_t-log_fx_t:
        X = np.vstack((X, y_t))
        x_t = np.copy(y_t)
        log_fx_t = log_fy_t
        count_variable[ii] +=1.0
        #n+=1
     delta = min(0.01, 1.0/(np.sqrt(batch)))
     sign = -1+2*(count_variable < NBatches*fraction_accepted)
     log_sigma += log_sigma + np.multiply(sign, delta)
     
    # if (t%100)==0:
    plt.plot(X[1:,3])
    plt.xlabel("Iteration")
    plt.ylabel(r"$x_1$")
    plt.savefig('x1_p3.eps', format='eps')
    plt.close()
  return X
def test_data(K):
 Y =  np.zeros((K, K))
 #serie = np.array([5, 50, 500, 5, 5, 5, 50, 50, 500, 500])
 serie = np.array([5, 50, 5, 5, 5, 5, 50, 50, 5, 5])
 R = np.tile(serie, int(K/len(serie)))
 for k1 in range(K):
  mu_k = k1*1.0
  sd_k = 10.0
  for r1 in range(R[k1]):
    Y[k1, r1] = norm.rvs(mu_k, sd_k**2)
 print(Y)
 return Y, R 
K = 50
Batchsize = 50
NBatches = 50000
Y, R = test_data(K)
X = MH(NBatches, Batchsize, Y, R)
 
#plt.plot(X[:,0])
#plt.ylim(0, 4)
#plt.savefig('destination_path.eps', format='eps')
#plt.show()

#xl, yl = np.mgrid[-30.0:30.0:.1, -30.0:30.0:.1]
#zl = np.copy(xl)
#for i in range(0, len(xl[0,:])):
#  for j in range(0, len(yl[0,:])):
#     zl[i,j] = f(np.array([xl[i,j], yl[i,j]]), B)
#
#plt.contourf(xl, yl, zl)
#plt.plot(X[:,0], X[:,1], 'r')
#plt.xlabel(r"$\alpha$")
#plt.ylabel(r"$\beta$")
#
#plt.show()



