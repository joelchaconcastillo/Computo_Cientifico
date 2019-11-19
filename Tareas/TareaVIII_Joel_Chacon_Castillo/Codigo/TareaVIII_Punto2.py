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
#np.random.seed(3)

def weib(x,n,a):
     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
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

def Fl2(x, ti, b, c, n):
  Alpha = x[0]
  Lambda = x[1]
#  p1 = (Alpha**n)*(Lambda**n)*( np.prod(np.power(ti,Alpha-1.0) ))*np.exp(-Lambda * np.sum( np.power(ti, Alpha) ))
#  p2 = c*np.exp(-Alpha*c)*Alpha
  #p3= ( Lambda**(Alpha-1))*np.exp(-b*Lambda)*(b**Alpha)
  #p1 =  np.prod(weibull_min.pdf(ti, Alpha, scale = 1.0/Lambda))
  p1 =  np.prod(weib(ti, Alpha, Lambda))
  p2 =  expon.pdf(Alpha, scale = 1.0/c) 
  p3 = gamma.pdf(Lambda, Alpha, scale=1.0/b)
  return p1 * p2 * p3 

###sampling proposals..
def q1(x, ti, n, b):##lambda step
 y = np.copy(x)
 sum_ti_alpha = np.sum(np.power(ti, y[0]))
 y[1] = gamma.rvs(y[0]+n, scale = 1.0/(b + sum_ti_alpha))
 return y
def q2(x, ti, n, b, c): ##alpha step
 y =np.copy(x)
 r1 = np.prod(ti)
 y[0] =  gamma.rvs(n+1, scale = 1.0/(-np.log(b) - np.log(r1) + c))
 return y
def q3(x, ti, b, c): ##both steps...
 alpha_p = expon.rvs(scale=c)
 Lambda_p = gamma.rvs(alpha_p, 1.0/b)
 return np.array([alpha_p, Lambda_p])
def q4(x):
  y = np.copy(x)
  y[0] += norm.rvs(0, 0.5)
  y[0] = max(0.01,  y[0])
  return y
######density_value..
def q1d(x, ti, n, b):##lambda step
 sum_ti_alpha = np.sum(np.power(ti, x[0]))
 return gamma.pdf(x[1], x[0]+n, scale = 1.0/(b + sum_ti_alpha))
def q2d(x, ti, n, b, c): ##alpha step
 r1 = np.prod(ti)
 return gamma.pdf(x[0], n+1, scale = 1.0/(-np.log(b) - np.log(r1) + c))
def q3d(x, ti, b, c): ##both steps...
 return gamma.pdf(x[0], 1.0/b)
def q4d(x):
  return norm.pdf( x[0], 0, 0.5)


def Hybrid_Gibbs(n, maxite, b, c, ti, burnin):
  xt = uniform.rvs(size=2)*4
  X = np.array([xt])
  FX = np.array([0])
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
   if np.any(yt) > 10 or np.any(yt) < 0.01: 
      continue

   fx = np.exp(Fl(xt, ti, b, c, n))
   fy = np.exp(Fl(yt, ti, b, c, n))
   fy_x = np.exp(Fl(yt, ti, b, c, n)- Fl(xt, ti, b, c, n))
   fy_x = (Fl2(yt, ti, b, c, n)/ Fl2(xt, ti, b, c, n))
   rho =min(1,(fy_x)*(dxt/dyt))
   u = (uniform.rvs())
   ##compute decision criteria
   if u < rho: ##accepting..
     if cont > burnin:
        X = np.vstack((X, yt))
        FX = np.vstack((FX, fy))
     xt = np.copy(yt)
     cont +=1
  return X, FX

def Punto2():
  n = 20
  Lambda = 1
  Alfa = 1
  b =1
  c = 1 
  maxite = 1000
  burnin = 100
  ## Generating ti's (data) from Weibull
  #ti = weibull_min.rvs(Alfa, loc=0, scale=1.0/Lambda, size=n)
  ti = np.random.weibull(Alfa, size=n)
  xd, fxd = Hybrid_Gibbs(n, maxite, b, c, ti, burnin)
#:  plt.plot(ti, weibull_min.pdf(ti, Alfa, loc=0, scale=Lambda), '.')
  plt.hist(xd, label='CDF',histtype='step', alpha=0.8, density=True )
  plt.title("Distribucion de los parámetros " r"$\alpha$ " "y" "$\lambda$")
  plt.legend([r"$\alpha$" ,r"$\beta$"])
  print(xd)
  plt.show()
  xl, yl = np.mgrid[0.1:3:.1, 0.1:3:.1]
  zl = np.copy(xl)
  for i in range(0, len(xl[0,:])):
    for j in range(0, len(xl[0,:])):
       zl[i,j] = (Fl2(np.array([xl[i,j], yl[i,j]]), ti, b, c, n))

  plt.contourf(xl, yl, zl)
#  plt.plot(xd[:,0], xd[:,1], '.r')
  plt.xlabel(r"$\alpha$")
  plt.ylabel(r"$\beta$")
  plt.show()
#  print(np.mean(xd, axis=0))
#
#  plt.plot(xd[:,0])
#  plt.title("Datos de la variable " r"$\alpha$")
#  plt.xlabel("Iteracion")
#  plt.ylabel("valor")
#  plt.show()
#
#  plt.plot(xd[:,1])
#  plt.title("Datos de la variable " r"$\beta$")
#  plt.xlabel("Iteracion")
#  plt.ylabel("valor")
#  plt.show()

Punto2()
