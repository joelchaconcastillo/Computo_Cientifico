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

###############################Section point 1

def qi(X, Mu, v_sigma, rho, i): #X_i | X_j
      j = 1-i
      Mu_i_j = Mu[i] + rho*(v_sigma[i]/v_sigma[j])*(X[j]-Mu[j])
      Sigma_i_j = (v_sigma[i]**2)*(1.0-rho**2)
      yt = np.copy(X)
      yt[i] = norm.rvs(Mu_i_j, Sigma_i_j)
      return yt
def energy(xt, yt, Mu, Sigma_Inv, v_sigma, rho, i):   #it is computed trough the logaritmic, X_j| x_i
    j = 1-i
    term1 = -0.5*np.dot(np.dot((yt- Mu), Sigma_Inv),yt-Mu)
    term2 = 0.5*np.dot(np.dot((xt- Mu), Sigma_Inv),xt-Mu)
    Mu_i_j_x = Mu[i] + rho*(v_sigma[i]/v_sigma[j])*(xt[j]-Mu[j])
    Mu_i_j_y = Mu[i] + rho*(v_sigma[i]/v_sigma[j])*(yt[j]-Mu[j])
    Sigma_i_j = (v_sigma[i]**2)*(1.0-rho**2)
    term3 = ((xt[i]-Mu_i_j_x)**2)/(2*(Sigma_i_j**2))
    term4 = ((xt[j]-Mu_i_j_y)**2)/(2*(Sigma_i_j**2))
    return term1+term2-term3+term4

def MH(maxite, Mu, Sigma, rho, v_sigma):
    Sigma_Inv = linalg.inv(Sigma) #precision matrix..
    U = (uniform.rvs(2)*2-1.0)*0.2
    xt = Mu+U# np.array([Mu[0]+uniform.rvs()*0.1, Mu[1]+uniform.rvs()*0.1])
    X = np.array([xt])
    cont = 0
    while cont < maxite:
       for k in range(2):
          d = np.random.randint(2)
          yt = qi(xt, Mu, v_sigma, rho, d)
	  ##check support ##it is not in the support
          if yt[d]==0: 
            continue	  
          criteria = energy(xt, yt, Mu, Sigma_Inv, v_sigma, rho, d)
          u = np.log(uniform.rvs())
	  ##compute decision criteria
          if u < criteria: ##accepting..
            X = np.vstack((X, yt))
            xt = np.copy(yt)
            cont +=1
    return X 
def Punto1():
   rho = 0.99
   v_sigma = np.array([1,1]) # sigma vector
   Mu = np.array([1, 1])
   Sigma = np.array([[v_sigma[0]**2, rho*v_sigma[0]*v_sigma[1]], [rho*v_sigma[0]*v_sigma[1], v_sigma[1]**2]])
   x = MH(10000, Mu, Sigma, rho, v_sigma)
   ##print contour ...
   xl, yl = np.mgrid[-1:3:.01, -1:3:.01]
   pos = np.empty(xl.shape + (2,))
   pos[:, :, 0] = xl
   pos[:, :, 1] = yl
   rv = multivariate_normal(Mu, Sigma)
   plt.contourf(xl, yl, rv.pdf(pos))
   plt.plot( x[:,0], x[:,1], '.r')
   plt.show()
########################################################Section point 2
###X[0] --> alpha
###X[1] --> lambda
def Fl(x, ti, b, c, n):
  Alpha = x[0]
  Lambda = x[1]
  p1 = n*np.log(Alpha) 
  p2 = n*np.log(Lambda)
  p3 = -Lambda*(np.sum(np.power(ti, Alpha)))
  p4 = -b*Lambda + c + Alpha*np.log(b) + (Alpha-1)*np.log(Lambda)
  return p1 + p2 + p3 + p4

###sampling proposals..
def q1(x, ti, n, b):##lambda step
 y = np.copy(x)
 sum_ti_alpha = np.sum(np.power(ti, y[0]))
 y[1] = gamma.rvs(y[0]+n, scale = b + sum_ti_alpha)
 return y
def q2(x, ti, n, b, c): ##alpha step
 y =np.copy(x)
 r1 = np.prod(ti)
 y[0] =  gamma.rvs(n+1, scale = -np.log(b) - np.log(r1) + c)
 return y
def q3(x, ti, b, c): ##both steps...
 alpha_p = expon.rvs(scale=c)
 Lambda_p = gamma.rvs(alpha_p, b)
 return np.array([alpha_p, Lambda_p])
def q4(x):
  y = np.copy(x)
  y[0] += norm.rvs(0, 0.5)
  y[0] = max(0.01,  y[0])
  return y
######density_value..
def q1d(x, ti, n, b):##lambda step
 sum_ti_alpha = np.sum(np.power(ti, x[0]))
 return gamma.pdf(x[1], x[0]+n, scale = b + sum_ti_alpha)
def q2d(x, ti, n, b, c): ##alpha step
 r1 = np.prod(ti)
 return gamma.pdf(x[0], n+1, scale = -np.log(b) - np.log(r1) + c)
def q3d(x, ti, b, c): ##both steps...
 return gamma.pdf(x[0], b)
def q4d(x):
  return norm.pdf( x[0], 0, 0.5)


def Hybrid_Gibbs(n, maxite, b, c, ti, burnin):
  xt = uniform.rvs(size=2)*4
  X = np.array([xt])
  yt = xt
  dyt = 0
  dxt = 0
  cont = 0
  while cont < maxite:
   d = np.random.randint(4) ##same probability of each kernel...
   if d == 0:
      yt = q1(xt, ti, n, b)
      dyt = q1d(xt, ti, n, b)
      dxt = q1d(yt, ti, n, b)
   elif d == 1:
      yt = q2(xt, ti, n, b, c)
      dyt = q2d(xt, ti, n, b, c)
      dxt = q2d(yt, ti, n, b, c)
   elif d == 2:
      yt = q3(xt, ti, b, c)
      dyt = q3d(xt, ti, b, c)
      dxt = q3d(yt, ti, b, c)
   elif d == 3:
      yt = q4(xt)
      dyt = q4d(xt)
      dxt = q4d(yt)
   ##check support ##it is not in the support
   if np.any(yt) > 10 or np.any(yt) < 0.1: 
      continue

   fx = np.exp(Fl(xt, ti, b, c, n))
   fy = np.exp(Fl(yt, ti, b, c, n))
   fy_x = np.exp(Fl(yt, ti, b, c, n)- Fl(xt, ti, b, c, n))
   rho =min(1,(fy_x)*(dxt/dyt))
   u = (uniform.rvs())
   ##compute decision criteria
   if u < rho: ##accepting..
     if burnin > cont:
        X = np.vstack((X, yt))
     xt = np.copy(yt)
     cont +=1
  return X 

def Punto2():
  n = 20
  Lambda = 1
  Alfa = 1
  b =1
  c = 1 
  maxite = 10000
  burnin = 500
  ## Generating ti's (data) from Weibull
  ti = weibull_min.rvs(Alfa, loc=0, scale=Lambda, size=n)
  xd = Hybrid_Gibbs(n, maxite, b, c, ti, burnin)
#:  plt.plot(ti, weibull_min.pdf(ti, Alfa, loc=0, scale=Lambda), '.')
  plt.hist(xd, label='CDF',histtype='step', alpha=0.8, density=True )
  print(xd)
  plt.show()
#########################################Section of point 3
def Punto3():

#Punto1()
#Punto2()
Punto3()
