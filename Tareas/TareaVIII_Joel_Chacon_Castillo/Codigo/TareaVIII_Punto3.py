import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm, norm, multivariate_normal, weibull_min, expon, loggamma
from scipy import stats, linalg
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(3)

#########################################Section of point 3
def factorial(n):
    return reduce((lambda x,y: x*y),range(1,n+1))
def posteriori(pi, ti, alpha, gamma_p, delta, lambdas_i, beta, fpi, n):
 # prod_lambda_t = np.multiply(lambdas_i, ti)
 # L1 = np.exp(-prod_lambda_t - lambdas_i*beta - delta*beta) 
 # L2 = np.power(prod_lambda_t, pi)/fpi
 # L3 = (beta**alpha)*(np.power(lambdas_i, alpha-1.0))*(delta**gamma_p)*(beta**(gamma_p-1.0))
 # return np.prod( np.multiply(L1, L2, L3))
  #prod_lambda_t = np.multiply(lambdas_i, ti)
  #L1 = np.exp(-prod_lambda_t)*np.power(prod_lambda_t, pi)/fpi
  #L2 = gamma.pdf(alpha, beta)
  #L3 = gamma.pdf(gamma_p, delta)
  #return np.prod(L1)*L2*L3
  L1 = np.sum(np.multiply(-ti, lambdas_i)) -beta*np.sum(lambdas_i-delta*beta)
  L2 = beta**(n*alpha - gamma_p - 1.0)
  L3 = np.prod( np.multiply(np.power(ti, pi),np.power(lambdas_i,pi+alpha-1))   )
  return L1*L2*L3

def q1_pump(t, p, i, alpha, x):
  y = np.copy(x)
  y[i] = gamma.rvs(p[i-1] + alpha, scale = 1.0/(t[i-1]+x[0]))
  return y
def q2_pump(x, i, n, alpha, gamma_p, delta):
  y = np.copy(x)
  sum_lambda = np.sum(x[1:])
  y[i] = gamma.rvs(n*alpha+gamma_p, scale = 1.0/(delta + sum_lambda))
  return y

def Hybrid_Gibbs_Pump_Failures(n, ti, pi, alpha, gamma_p, delta, maxite, burnin):
  xt = uniform.rvs(size=11)*0.02
  X = np.array([xt])
  fpi = np.copy(pi)
  for i in range(n):
   fpi[i] = math.factorial(pi[i]) 
  yt = xt
  cont = 0
  d=-1
  while cont < maxite:
   d=(d+1)%11
#   d = np.random.randint(11) ##same probability of each kernel...
   if d >=1:
      yt = q1_pump(ti, pi, d, alpha, xt)
   elif d == 0:
      yt = q2_pump(xt, d, n, alpha, gamma_p, delta)
   ##check support ##it is not in the support
#   fx = posteriori(pi, ti, alpha, gamma_p, delta, xt[1:], xt[0], fpi, n)
   ##compute decision criteria
   if burnin < cont:
     X = np.vstack((X, yt))
   if (cont%1000)==0:
     print(str(cont) + " "+str(xt))
   xt = np.copy(yt)
   cont +=1
  return X 


def Punto3():
 n = 10
 ti = np.array([94.32, 15.72, 62.88, 125.76, 5.24, 31.44, 1.05, 1.05, 2.1, 10.48])
 pi = np.array([5.0, 1.0, 5.0, 14.0, 3.0, 19.0, 1.0, 1.0, 4.0, 22.0])
 alpha = 1.8
 gamma_p = 0.01
 delta = 1.0
 burnin = 1000
 maxite = 10000
 x = Hybrid_Gibbs_Pump_Failures(n, ti, pi, alpha, gamma_p, delta, maxite, burnin)

 print(np.mean(x, axis=0))
 print("k")
# for i in range(1,11):
#     plt.plot(x[:,i])
#     plt.title("Datos de la variable " r"$\lambda_"+str(i)+"$")
#     plt.show()
 plt.plot(x[:,0])
 plt.title("Datos de la variable " r"$\beta")
 plt.show()


Punto3()
