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


def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def log_f1(Y, R, x):
 global mu_p, sigma_p, a1, a2, b1, b2;
 K = len(Y[:,0])
 mu = x[0]
 A =  np.exp(x[1])
 V = np.exp(x[2])
 #sum = -((mu-mu_p)**2)/(2.0*(sigma_p**2))
 #sum += -b1/A - (a1+1.0)*np.log(A)
 #sum += -b2/V - (a2+1.0)*np.log(V)
 #sum += np.log(A)+np.log(V)
 
 sum= -(mu-mu_p)*(mu-mu_p) / 2.0 / sigma_p / sigma_p -b1/A - (a1+1) * np.log(A) -b2/V - (a2+1) * np.log(V) + np.log(A) + np.log(V);

 for kk in range(K):
  #print(kk)
  sum += log_f2(Y, R, x, kk)
 return sum

def log_f2(Y, R, x, k2):
 global mu_p, sigma_p, a1, a2, b1, b2;
 mu = x[0]
 A = np.exp(x[1])
 V = np.exp(x[2])
 sum = -np.log(A) - ((x[k2+3]-mu)/A)**2
 #sum = -np.log(1.0 + ((x[k2]-mu)/A)**2)
 #sum += np.sum(-np.power(  Y[k2-3, 0:R[k2-3]] - x[k2], 2)/(2.0*V**2)) #- 0.5*np.log(V)
 #sum = - np.log(A) - ( (x[k2+3]-mu) / A )**2;
 for jj in range(R[k2]):
  sum= sum- (Y[k2][jj]-x[k2+3])**2 / 2.0 / V - 0.5 * np.log(V);
 return sum


def MH(NBatches, Batchsize, Y, R, adaptive):
  K = len(Y[0,:])
  x_t = np.zeros(K+3) ## theta_i, mu, A, V
  #x_t[0:3]=1.0
  log_sigma = np.zeros(K+3)
  M_log_sigma = np.copy(log_sigma)
  X = np.copy(x_t)
  log_fx_t =  log_fy_t = 0
  sigma = 1
  n = 0
  t = 0.0
  fraction_accepted = 0.44
#  while t < maxite:
  for t in range(1, NBatches):
    count_variable= np.zeros(K+3)
    for batch in range(1,Batchsize):
     y_t = np.copy(x_t)
#     print(batch)
     for ii in range(K+3):
      y_t[ii] = x_t[ii] + np.exp(log_sigma[ii])*(norm.rvs(0,1.0))
      #y_t[ii] = norm.rvs(x_t[ii],np.exp(log_sigma[ii])**2)
      if ii < 3:
          log_fy_t = log_f1(Y, R, y_t)
          log_fx_t = log_f1(Y, R, x_t)
      else:
          log_fy_t = log_f2(Y, R, y_t, ii-3)
          log_fx_t = log_f2(Y, R, x_t, ii-3)
      if np.log(uniform.rvs(0.0, 1.0)) < log_fy_t-log_fx_t:
        #X = np.vstack((X, y_t))
        x_t[ii] = y_t[ii]
        log_fx_t = log_fy_t
        count_variable[ii] +=1.0
      else:
        y_t[ii] = x_t[ii]
     if adaptive:
      delta = min(0.01, 1.0/(np.sqrt(t)))
      sign = -1.0+2.0*(count_variable > Batchsize*fraction_accepted)
      log_sigma += (np.multiply(sign, delta))
     #log_sigma = log_sigma.clip(min=0.0)
     #print((log_sigma))
    print("Batch "+str(t))
    X = np.vstack((X, y_t))
    M_log_sigma = np.vstack((M_log_sigma, np.exp(log_sigma)))
    if (t%50)==0 and adaptive:
     for th in range(3,6):
       plt.plot(X[1:,th])
       plt.xlabel("Batch Iteration")
       plt.ylabel(r"$\theta_"+str(th-2)+"$")
       plt.savefig('theta_'+str(th-2)+'_p3.eps', format='eps')
       plt.close()
     for lsi in range(3,6):
       plt.plot((M_log_sigma[1:,lsi]))
       plt.xlabel("Batch Iteration")
       plt.ylabel(r"$ls_"+str(lsi-2)+"$")
       plt.savefig('ls'+str(lsi-2)+'_p3.eps', format='eps')
       plt.close()
     for i in range(3,9):
      print(np.mean((X[1:,i]- X[:-1,i])**2))

  return X
def test_data(K):
 Y =  np.zeros((K, K))
 #serie = np.array([5, 50, 500, 5, 5, 5, 50, 50, 500, 500])
 serie = np.array([3, 2, 3, 2, 2, 3, 3, 1, 3, 2])
 R = np.tile(serie, int(K/len(serie)))
 for k1 in range(K):
  mu_k = k1*1.0
  sd_k = 10.0
  for r1 in range(R[k1]):
    Y[k1, r1] = norm.rvs(mu_k, sd_k**2)
 #*print(Y)
 return Y, R 
K = 10
Batchsize = 50
NBatches = 5000
adaptive = False
Y, R = test_data(K)
X = MH(NBatches, Batchsize, Y, R, adaptive)

for i in range(3,9):
# print(np.mean(estimated_autocorrelation(X[:,i])))
 print(np.mean((X[1:,i]- X[:-1,i])**2))
 
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



