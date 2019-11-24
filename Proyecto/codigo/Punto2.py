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

def log_f(x, B):
 sum = - (x[0]**2)/200.0 - 0.5*(x[1] + B*(x[0]**2) - 100.0*B)**2
 if len(x) > 2:
   sum += np.sum( np.power(x[2:], 2))
 return sum
def f(x, B):
 sum = - (x[0]**2)/200.0 - 0.5*(x[1] + B*(x[0]**2) - 100.0*B)**2
 if len(x) > 2:
   sum += np.sum( np.power(x[2:], 2))
 return np.exp(sum)

def MH(maxite, d, B):
  x_t = np.zeros(d)
  B = 0.1
#  for i in range(d):
#    x_t[i] =  uniform.rvs(0.01) 
  X_t = np.zeros((d,d))
  log_fx_t =  log_f(x_t, B)#multivariate_normal.logpdf(x_t, np.ones(d), Cov)
  n = 0
  beta = 0.05
  X = np.copy(x_t)
  X2 = np.copy(x_t)
  mix_cov = (0.1**2)*np.ones(d)/(float(d))
  cov_sums = np.zeros((d,d))
  meanx = np.copy(x_t)
  t = 0.0
  b = np.zeros(1)
  while t < maxite:
     t += 1.0
     idx = choice([0,1],p= [1.0-beta, beta])
     if n <= 2*d or idx ==1:
        y_t = multivariate_normal.rvs(x_t, mix_cov) ##instead it could be something like   y_t = x_t + base_mix_cov*multivariate_normal.rvs(0.0, np.ones((d,d))); base_mix could be decomposition cholesky
     else:
        if idx == 0:
           empirical_cov = cov_sums/(t-d)
           Covp = empirical_cov#np.cov(X2.T)
      #     Covp = np.cov(X2.T) ###instead of incremental variance....
           Covp2 = (2.381204**2)*Covp/(float(d))
           y_t = multivariate_normal.rvs(x_t, Covp2)
     log_fy_t = log_f(y_t, B) #multivariate_normal.logpdf(y_t, np.zeros(d), Cov)
     if np.log(uniform.rvs(0.0, 1.0)) < log_fy_t-log_fx_t:
       X = np.vstack((X, y_t))
       x_t = np.copy(y_t)
       log_fx_t = log_fy_t
       #n+=1
       if (t%10000)==0:
         plt.plot(X[1:,0])
         plt.xlabel("Iteration")
         plt.ylabel(r"$x_1$")
         plt.savefig('x1.eps', format='eps')
         plt.close()
     #X2 = np.vstack((X2, x_t))
     ##Incremental mean and incremental variance....
     oldmeanx = np.copy(meanx)
     meanx = (t/(t+1.0))*meanx+ (1.0/(t+1.0))*x_t
     diff1 = oldmeanx-meanx
     diff2 = x_t - meanx
     cov_sums += (t-1.0)*np.outer(diff1, diff1) + np.outer(diff2,diff2)
     ######ratio bound...

  return X
    
maxite = 10000
d = 2 #dimension...
#print(Cov)
B = 0.1
X = MH(maxite, d, B)

#plt.plot(X[:,0])
#plt.ylim(0, 4)
#plt.savefig('destination_path.eps', format='eps')
#plt.show()

xl, yl = np.mgrid[-30.0:30.0:.1, -30.0:30.0:.1]
zl = np.copy(xl)
for i in range(0, len(xl[0,:])):
  for j in range(0, len(yl[0,:])):
     zl[i,j] = f(np.array([xl[i,j], yl[i,j]]), B)

plt.contourf(xl, yl, zl)
plt.plot(X[:,0], X[:,1], 'r')
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$")

plt.show()



