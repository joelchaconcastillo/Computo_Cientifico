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

   
    
def MH(maxite, d, Cov):
  x_t = np.zeros(d)
  for i in range(d):
    x_t[i] =  uniform.rvs(0.01) 
  log_fx_t =  multivariate_normal.logpdf(x_t, np.ones(d), Cov)
  n = 0
  beta = 0.05
  X = np.copy(x_t)
  X2 = np.copy(x_t)
  mix_cov = (0.1**2)*np.ones(d)/(float(d))
  cov_sums = np.zeros((d,d))
  meanx = np.copy(x_t)
  t = 0.0
  b = np.zeros(1)
  while n < maxite:
     t += 1.0
     if n <= 2*d:
        y_t = multivariate_normal.rvs(x_t, mix_cov)
     else:
        idx = choice([0,1],p= [1.0-beta, beta])
        if idx == 0:
           empirical_cov = cov_sums/(t-d)
           Covp = empirical_cov#np.cov(X2.T)
      #     Covp = np.cov(X2.T)
           Covp2 = (2.381204**2)*Covp/(float(d))
           y_t = multivariate_normal.rvs(x_t, Covp2)
        elif idx ==1:
           y_t = multivariate_normal.rvs(x_t, mix_cov)
     log_fy_t = multivariate_normal.logpdf(y_t, np.zeros(d), Cov)
     if np.log(uniform.rvs(0.0, 1.0)) < log_fy_t-log_fx_t:
       X = np.vstack((X, y_t))
       x_t = y_t
       log_fx_t = log_fy_t
       n+=1
       if (n%100)==0:
         plt.plot(X[1:,0])
         plt.xlabel("Iteration")
         plt.ylabel(r"$x_1$")
#         plt.ylim(0, 5)
         plt.savefig('x1.eps', format='eps')
         plt.close()

       print(n)
     #X2 = np.vstack((X2, x_t))
     ##Incremental mean and incremental variance....
     oldmeanx = np.copy(meanx)
     meanx = (t/(t+1.0))*meanx+ (1.0/(t+1.0))*x_t
     diff1 = oldmeanx-meanx
     diff2 = x_t - meanx
     cov_sums += (t-1.0)*np.outer(diff1, diff1) + np.outer(diff2,diff2)
     ######ratio bound...
#     print(empirical_cov)
     if n > d:#2*d:
       empirical_cov =  cov_sums/(t-d)
       #empirical_cov = np.cov(X2.T)
       cov1 = scipy.linalg.sqrtm(empirical_cov)
       cov2_inv = np.linalg.inv(scipy.linalg.sqrtm(Cov))
       eig = np.linalg.eigvals(cov1.dot(cov2_inv) + np.ones((d,d))*1e-5)
       v = d*(np.sum(np.power(eig, -2))/ (np.sum(1.0/eig)**2))
       b = np.vstack((b,v))
       if (n%100)==0:
         plt.plot(b[1:])
         plt.xlabel("Iteration")
         plt.ylabel("Suboptimality factor "+r"$b$")
         plt.ylim(0, 5)
         plt.savefig('optimal.eps', format='eps')
         plt.close()

  return X,b[1:]
    
maxite = 50000
d = 10 #dimension...
M = np.zeros((d, d))
for i in range(d):
  for j in range(d):
    M[i,j] = norm.rvs(0.0, 1.0)
Cov = M.dot(M.T)
#print(Cov)
X,b = MH(maxite, d, Cov)

#plt.plot(X[:,0])
plt.plot(b)
plt.ylim(0, 4)
plt.savefig('destination_path.eps', format='eps')
#plt.show()

#xl, yl = np.mgrid[-10.0:10.0:.1, -10.0:10.0:.1]
#zl = np.copy(xl)
#for i in range(0, len(xl[0,:])):
#  for j in range(0, len(yl[0,:])):
#     zl[i,j] = multivariate_normal.pdf(np.array([xl[i,j], yl[i,j]]), np.zeros(d), Cov)
#
#plt.contourf(xl, yl, zl)
#plt.plot(X[:,0], X[:,1], 'r')
#plt.xlabel(r"$\alpha$")
#plt.ylabel(r"$\beta$")
#
#plt.show()
#


